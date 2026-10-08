"""
TTOA-NN Framework: Taboo Trade-Off Aversion Neural Network

The model separates alternative-specific utility from shared cost utility using
the ASS-NN approach as described in Hernández et al. (2023). The TTOA-NN framework
adds a TTOA effect to the ASS-NN model, which can be homogeneous, conditioned on
respondent covariates, or conditioned on both covariates and text-derived features.

Reference:
* Hernández JI, Mouter N, and van Cranenburgh S. (2023). An economically-consistent
discrete choice model with flexible utility specification based on artificial neural
networks. ArXiv preprint arXiv:2404.13198.

"""

import torch
from torch import nn
import torch.nn.functional as F

class TTOA_NN(nn.Module):
    """
    TTOA-NN embedded into the ASS-NN model for discrete choice modeling.

    * asu_layers: models alternative-specific non-cost attributes
    * cost_layers: models shared cost attributes between alternatives in a choice set
    * taboo_layers: taboo degree learner of trade-offs between attributes across alternatives
    * ttoa_layers: models the TTOA effect modulation based on covariates and/or text features
    * z_layers: models covariate-based features for TTOA effect modulation
    * text_layers: models text-derived features for TTOA effect modulation
    * moral_stance_layers: prediction head for moral stance scores

    """

    @staticmethod
    def _activation(name):
        """Return the configured hidden-layer activation function"""
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        try:
            return activations[name]()
        except KeyError as error:
            raise ValueError(
                "Unsupported activation function. Choose from 'relu', "
                "'tanh', or 'sigmoid'."
            ) from error

    @classmethod
    def _build_mlp(cls, layer_sizes, prefix, activ_fn, dropout=0.0):
        """Build a MLP while retaining the layer names"""
        layers = nn.Sequential()
        for layer_index, (in_size, out_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:]), start=1
        ):
            layers.add_module(
                name=f"{prefix}_Layer_{layer_index}",
                module=nn.Linear(in_size, out_size, bias=True),
            )
            is_hidden_layer = layer_index < len(layer_sizes) - 1
            if is_hidden_layer:
                layers.add_module(
                    name=f"{prefix}_Activ_{layer_index}",
                    module=cls._activation(activ_fn),
                )
                if dropout > 0:
                    layers.add_module(
                        name=f"{prefix}_Dropout_{layer_index}",
                        module=nn.Dropout(p=dropout),
                    )
        return layers

    def __init__(self, asu_layer_sizes, cost_layer_sizes, taboo_layer_sizes=None,
                 z_layer_sizes=None, text_layer_sizes=None, ttoa_layer_sizes=None,
                 moral_stance_layer_sizes=None, ttoa_model=False, z_model=False,
                 text_model=False, activ_fn="relu", dropout=0):

        super().__init__()

        # Set global parameters
        self.ttoa_model = ttoa_model
        self.z_model = z_model
        self.text_model = text_model

        # ASS-NN backbone: alternative-specific non-cost and shared cost paths.
        self.asu_layers = self._build_mlp(asu_layer_sizes, "ASU", activ_fn)
        self.cost_layers = self._build_mlp(cost_layer_sizes, "Cost", activ_fn)

        #------------------------------------------------------------------
        # The TTOA-NN Framework: Taboo degree learner of trade-offs
        #------------------------------------------------------------------

        if self.ttoa_model:
            self.taboo_layers = self._build_mlp(taboo_layer_sizes, "Taboo", activ_fn)
                    
        #------------------------------------
        # Sub-network: TTOA effect modulation
        #------------------------------------

            if self.z_model is False:
                # Define lambda TTOA as a single parameter (homogeneous effect)
                self.ttoa = nn.Parameter(torch.tensor(0.0))

            elif self.z_model is True:
                #------------------------------------
                # Module: z variables for lambda TTOA
                #------------------------------------

                self.z_layers = self._build_mlp(z_layer_sizes, "Z", activ_fn)
                        
                #---------------------------------------
                # Module: text documents for lambda TTOA
                #---------------------------------------
                
                if text_layer_sizes is not None:
                    self.text_layers = self._build_mlp(text_layer_sizes, "Text", activ_fn)
                
                #-------------------------------------------------------------------------
                # Head: interaction between z variables and text documents for lambda TTOA
                #-------------------------------------------------------------------------

                    self.ttoa_layers = self._build_mlp(
                        ttoa_layer_sizes, "ttoa", activ_fn, dropout
                    )

                #--------------------------------------------------
                # Head: moral stance scores based on text documents
                #--------------------------------------------------

                    if moral_stance_layer_sizes is not None:
                        self.moral_stance_layers = self._build_mlp(
                            moral_stance_layer_sizes,
                            "Moral_stance",
                            activ_fn,
                            dropout,
                        )

    def forward(self, x_noncost, x_cost, x_taboo=None, z=None, text=None, text_mask=None):
        """
        Forward pass of the model

        Parameters:
        * x_noncost = dictionary of alternative-specific non-cost attributes
        * x_cost = dictionary of shared cost attributes
        * x_taboo = tensor of pairwise taboo trade-off values
        * z = tensor of individual-specific covariates for TTOA effect modulation
        * text = tensor of text features for TTOA effect modulation
        * text_mask = tensor of masks for moral stance prediction

        Inputs required only by the enabled TTOA-NN model:
        * x_taboo is required if ttoa_model is True
        * z is required if z_model is True
        * text is required if text_model is True
        * text_mask is required if text_model is True

        Returns:
        * choice = tensor of utilities for each alternative in each choice set
        * moral = tensor of predicted moral stance scores (if text_mask is provided)

        """

        #-------------------------------------------------------------------------
        # The cost layers are shared across alternatives, but evaluated separately
        # for each alternative as they have their own cost input.
        #-------------------------------------------------------------------------

        cost_out = {}

        for alt_name in sorted(x_cost.keys()):
            cost_out[alt_name] = self.cost_layers(x_cost[alt_name])

        if self.ttoa_model is True:
            taboo_out = self.taboo_layers(x_taboo)
            taboo_out = -torch.sigmoid(taboo_out)

            if self.z_model is True:
                z_out = self.z_layers(z)

                if self.text_model is False:
                    ttoa_out = F.softplus(z_out)

                if self.text_model is True:
                    text_out = self.text_layers(text)

                    moral_concat = torch.cat([F.layer_norm(z_out, z_out.shape[1:]),
                                              F.layer_norm(text_out, text_out.shape[1:])], dim=-1)
                    
                    ttoa_out = self.ttoa_layers(moral_concat)
                    ttoa_out = F.softplus(ttoa_out)

                    if text_mask is not None:
                        moral_out = self.moral_stance_layers(moral_concat) * text_mask

            elif self.z_model is False:
                ttoa_out = F.softplus(self.ttoa)
        else:
            ttoa_out = None
        
        V_output = {}

        for alt_name in sorted(x_noncost.keys()):
            asu_out = self.asu_layers(x_noncost[alt_name])

            # Utility without TTOA effect
            V = asu_out + cost_out[alt_name]

            # Add TTOA effect modulation to utility
            if self.ttoa_model is True:
                if alt_name == "alt1":
                    V = V + ttoa_out * taboo_out
                elif alt_name == "alt2":
                    V = V
                
            V_output[alt_name] = V.sum(dim=1, keepdim=True)
            
        V_mat = torch.cat([V_output["alt1"], V_output["alt2"]], dim=1)

        if text_mask is not None:
            return {"choice": V_mat, "moral": moral_out}
        else:
            return {"choice": V_mat}