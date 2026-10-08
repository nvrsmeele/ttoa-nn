############################################################
# Dutch pandemic preparedness experiment
# Estimating the TTOA-MNL model.
# Two alternatives: A1 and A2.
############################################################

rm(list = ls())

### Set working directory for R initialization
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

### Load library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "TTOA-MNL",
  modelDescr      = "TTOA-MNL model",
  indivID         = "RespID",
  outputDirectory = paste(getwd(),'results',"ttoa-mnl",sep=.Platform$file.sep),
  panlData = TRUE
)

### Load data
path_data = paste(getwd(),'data',"data.csv",sep=.Platform$file.sep)
database = read.csv(path_data, header=TRUE)

### Initialise model params
apollo_beta=c(b_deaths     = 0,
              b_pinjury    = 0,
              b_minjury    = 0,
              b_phealth    = 0,
              b_spressure1 = 0,
              b_spressure2 = 0,
              b_spressure3 = 0,
              b_tax        = 0,
              b_taboo      = 0)

### Fixed params: should be in quotes (optional)
apollo_fixed = c()

### Checkpoint for model inputs
apollo_inputs = apollo_validateInputs()

### Define model and likelihood function
apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
    
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  ### Create list of choice probabilities P
  P = list()
  
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[["A1"]]  = b_deaths * alt1_deaths + b_pinjury * alt1_pinjury + b_minjury * alt1_minjury + b_phealth * alt1_phealth + b_spressure1 * alt1_spressure_1 + b_spressure2 * alt1_spressure_2 + b_spressure3 * alt1_spressure_3 + b_tax * alt1_tax #+ b_taboo * alt1_taboo
  V[["A2"]]  = b_deaths * alt2_deaths + b_pinjury * alt2_pinjury + b_minjury * alt2_minjury + b_phealth * alt2_phealth + b_spressure1 * alt2_spressure_1 + b_spressure2 * alt2_spressure_2 + b_spressure3 * alt2_spressure_3 + b_tax * alt2_tax #+ b_taboo * alt2_taboo
  
  ### Initialise settings for MNL model component
  mnl_settings = list(
    alternatives  = c(A1=0, A2=1), 
    avail         = list(A1=1, A2=1), 
    choiceVar     = Choice,
    utilities     = V
  )
  
  ### Compute choice probabilities using MNL model
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

#### Model estimation
model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

### Print model output with p-values
modelOutput_setting=list(printPVal=2)
apollo_modelOutput(model, modelOutput_setting)

### Save model output with p-values
apollo_saveOutput(model, modelOutput_setting)

########################################################################################
#  Out-of-sample evaluation
########################################################################################

# Load test data
path_test_data = paste(getwd(),'data',"test.csv",sep=.Platform$file.sep)
database = read.csv(path_test_data, header=TRUE)

# Validate inputs
apollo_inputs_test = apollo_validateInputs()

# Compute predicted probabilities
P_test = apollo_probabilities(
  apollo_beta   = model$estimate,
  apollo_inputs = apollo_inputs_test,
  functionality = "prediction"
)

# Compute log-likelihood
Choice = database$Choice

# Convert probabilities to numeric vectors
A1_probs = as.numeric(unlist(P_test$model$A1))
A2_probs = as.numeric(unlist(P_test$model$A2))

# Get probability of chosen alternative
P_chosen = ifelse(Choice == 0, A1_probs, A2_probs)

# Log-likelihood
LL_test = sum(log(P_chosen))
cat("Log-likelihood:", LL_test, "\n")

# Null model (empirical shares)
share_A1 = mean(Choice == 0)
share_A2 = 1 - share_A1

LL_null = sum(ifelse(Choice == 0, log(share_A1), log(share_A2)))

# McFadden rho-squared
rho2_test = 1 - (LL_test / LL_null)
cat("Rho-squared:", rho2_test, "\n")