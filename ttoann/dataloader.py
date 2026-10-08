"""
Dataloader for the TTOA-NN model

Requirement:
* Choice data (csv file) in wide format, one row per choice observation

Note:
This module converts the wide-format csv file to tensors once during
dataset construction so that ``__getitem__`` only assembles the sample
required by the model.

"""

import pandas as pd
import torch
from torch.utils.data import Dataset

class ChoiceData(Dataset):
    """
    Load choice observations into a tensor-based dataset.

    Inputs:
    * data_path: Path to a wide-format CSV file.
    * model_type:
        - "multitask" includes the five moral-stance targets
        - "singletask" only includes the choice target
    * model_pred:
        - "choice" loads the alternative-specific attribute inputs
        - "choice-ttoa-z" adds taboo attributes and respondent covariates
        - "choice-ttoa-full" also adds PCA text features and an informative-
          document mask

    """

    #-----------------------------------------------
    # Define global variables for the dataset class
    #-----------------------------------------------

    supported_model_types = {"multitask", "singletask"}
    supported_predictions = {"choice", "choice-ttoa-z", "choice-ttoa-full"}

    cost_columns = {
        "alt1": ("alt1_tax",),
        "alt2": ("alt2_tax",),
    }

    noncost_columns = {
        "alt1": (
            "alt1_deaths", "alt1_pinjury", "alt1_minjury", "alt1_phealth",
            "alt1_spressure_1", "alt1_spressure_2", "alt1_spressure_3",
        ),
        "alt2": (
            "alt2_deaths", "alt2_pinjury", "alt2_minjury", "alt2_phealth",
            "alt2_spressure_1", "alt2_spressure_2", "alt2_spressure_3",
        ),
    }

    taboo_columns = (
        "delta_death", "delta_pinjury", "delta_minjury", "delta_phealth",
        "delta_spressure", "delta_tax",
    )

    z_columns = {
        "gender": "geslacht", "age": "leeftijd", "partner": "partner",
        "educ_2": "oplcat_2", "educ_3": "oplcat_3", "educ_4": "oplcat_4",
        "educ_5": "oplcat_5", "educ_6": "oplcat_6", "sted_2": "sted_2",
        "sted_3": "sted_3", "sted_4": "sted_4", "sted_5": "sted_5",
        "woning_1": "woning_1", "woning_2": "woning_2",
        "aantalhh": "aantalhh", "aantalki": "aantalki",
    }

    text_columns = tuple(f"pca_{index}" for index in range(1, 31))
    moral_columns = ("care", "fairness", "loyalty", "authority", "sanctity")

    #-----------------------------------------------
    # Initialize the dataset class methods
    #-----------------------------------------------

    def __init__(self, data_path, model_type="multitask", model_pred=None):
        if model_type not in self.supported_model_types:
            raise ValueError(f"model_type must be one of {self.supported_model_types}")
        if model_pred not in self.supported_predictions:
            raise ValueError(f"model_pred must be one of {self.supported_predictions}")

        self.model_type = model_type
        self.model_pred = model_pred
        self.data = pd.read_csv(data_path)
        self.n_samples = len(self.data)

        print(f"===> Initializing {model_pred} {model_type} dataset...")

        # Keep identifiers as integers for post hoc respondent-level analysis
        self.respid = self._column_tensor("RespID", torch.long)

        # These inputs are always required by the ASS-NN choice model
        self.cost_attr = self._nested_tensors(self.cost_columns)
        self.noncost_attr = self._nested_tensors(self.noncost_columns)

        if self.model_pred != "choice":
            self.taboo_attr = self._tensor_dict(self.taboo_columns)
            self.taboo_attr_list = self.taboo_columns
            self.z = self._tensor_dict(self.z_columns)

        if self.model_pred == "choice-ttoa-full":
            self.text = self._tensor_dict(self.text_columns)
            self.text_mask = self._column_tensor("Informative").unsqueeze(1)

        self.y = self._column_tensor("Choice", torch.long)

        if self.model_type == "multitask":
            self.y_2 = self._tensor_dict(self.moral_columns) # assuming "y" = "Choice"
            self.moral_dims = self.moral_columns

        print("===> Dataset initialized!")

    def _column_tensor(self, column, dtype=torch.float):
        """Convert one CSV column to a one-dimensional tensor."""
        return torch.as_tensor(self.data[column].to_numpy(), dtype=dtype)

    def _tensor_dict(self, columns):
        """Convert named columns while retaining their declared order."""
        if isinstance(columns, dict):
            return {name: self._column_tensor(column) for name, column in columns.items()}
        return {column: self._column_tensor(column) for column in columns}

    def _nested_tensors(self, columns_by_alternative):
        """Convert alternative-specific columns to nested dictionaries."""
        return {
            alternative: self._tensor_dict(columns)
            for alternative, columns in columns_by_alternative.items()
        }

    #--------------------------------------------------
    # Sample observations for the model's forward pass
    #--------------------------------------------------

    def __getitem__(self, idx):
        """Return observations required by the model's forward pass"""
        sample = {
            "respid": self.respid[idx],
            "x_noncost": self._stack_nested(self.noncost_attr, idx),
            "x_cost": self._stack_nested(self.cost_attr, idx),
            "y": self.y[idx],
        }

        if self.model_pred == "choice":
            return sample

        sample.update({
            "x_taboo": self._stack_named(self.taboo_attr, self.taboo_attr_list, idx),
            "z": self._stack_named(self.z, self.z.keys(), idx),
        })

        if self.model_pred == "choice-ttoa-z":
            return sample

        sample.update({
            "text": self._stack_named(self.text, self.text.keys(), idx),
            "text_mask": self.text_mask[idx],
        })
        if self.model_type == "multitask":
            sample["y_2"] = self._stack_named(self.y_2, self.moral_dims, idx)
        return sample

    @staticmethod
    def _stack_named(tensors, names, idx):
        """Select one value from each named tensor and stack the values."""
        return torch.stack([tensors[name][idx] for name in names])

    @classmethod
    def _stack_nested(cls, tensors_by_alternative, idx):
        """Stack each alternative's attributes without changing key order."""
        return {
            alternative: cls._stack_named(attributes, attributes.keys(), idx)
            for alternative, attributes in tensors_by_alternative.items()
        }
    
    def __len__(self):
        """Return the number of observations in the CSV file."""
        return self.n_samples
