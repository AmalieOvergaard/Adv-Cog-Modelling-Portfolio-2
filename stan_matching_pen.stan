data {
  int<lower=1> T;  
  // Number of trials in the task

  array[T] int<lower=0, upper=1> Self;  
  // The agent's observed choices on each trial (0 or 1)

  array[T] int<lower=0, upper=1> Other; 
  // The opponent's observed choices on each trial (0 or 1)
}

parameters {
  real<lower=0, upper=1> alpha;   
  // Learning rate: controls how strongly new opponent choices update memory

  real<lower=0> beta;             
  // Choice sensitivity: controls how strongly memory influences choice probability

  real bias;                      
  // General side bias: captures a tendency to prefer one response side

  real<lower=0, upper=1> m0;      
  // Initial memory state before the first trial
}

transformed parameters {
  vector[T] eta;            
  // Linear predictor (log-odds of choosing 1) for each trial

  vector[T] p;              
  // Choice probability of choosing 1 on each trial

  vector[T] memory_before;  
  // Memory state before observing the opponent on each trial

  vector[T] memory_after;   
  // Memory state after updating from the opponent's choice

  real m_now;               
  // Temporary variable storing the current memory state as the model moves through trials

  m_now = m0;               
  // Initialize memory at the starting value m0

  for (t in 1:T) {
    memory_before[t] = m_now;
    // Store the current memory before making the update on this trial

    eta[t] = bias + beta * (2 * (m_now - 0.5));
    // Compute the log-odds of choosing 1
    // Memory is centered around 0.5, so values above 0.5 favor choosing 1
    // and values below 0.5 favor choosing 0

    p[t] = inv_logit(eta[t]);
    // Convert log-odds into a probability between 0 and 1

    m_now = m_now + alpha * (Other[t] - m_now);
    // Update memory using an exponential / Rescorla-Wagner style learning rule
    // Prediction error = observed opponent choice - current memory

    memory_after[t] = m_now;
    // Store the updated memory after observing the opponent's choice
  }
}

model {
  alpha ~ beta(2, 2);
  //alpha ~ beta(3, 3);
  // Prior for learning rate:
  // keeps alpha between 0 and 1 and mildly favors middle values over extremes

  beta ~ lognormal(0, 0.5);
  //beta ~ lognormal(0, 0.3);
  // Prior for sensitivity:
  // ensures beta is positive and favors moderate values

  bias ~ normal(0, 1);
  // Prior for side bias:
  // centered at no bias, while allowing moderate positive or negative bias

  m0 ~ beta(2, 2);
  // Prior for initial memory:
  // keeps m0 between 0 and 1 and favors starting beliefs near the middle

  for (t in 1:T) {
    Self[t] ~ bernoulli_logit(eta[t]);
    // Likelihood:
    // the observed choice on each trial is modeled as a Bernoulli draw
    // with log-odds given by eta[t]
  }
}

generated quantities {
  array[T] int y_rep;  
  // Posterior predictive choices simulated from the fitted model

  vector[T] log_lik;   
  // Log-likelihood of each observed trial, useful for model comparison or diagnostics

  for (t in 1:T) {
    y_rep[t] = bernoulli_logit_rng(eta[t]);
    // Generate one replicated choice for each trial from the fitted model

    log_lik[t] = bernoulli_logit_lpmf(Self[t] | eta[t]);
    // Store the pointwise log-likelihood of the observed choice
  }
}

