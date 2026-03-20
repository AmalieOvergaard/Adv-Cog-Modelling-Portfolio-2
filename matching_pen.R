# --- MATCHING PENNIES --- 

# Setup
set.seed(1999)
library(tidyverse)
library(cmdstanr)
library(ggplot2)

# Opponent: chooses 1 with probability p_one (biased random agent)
# this opponent has no learning, strategy - it just tends to choose 1 more often (0.6)
simulate_opponent <- function(trials, p_one = 0.6) { # defines function simulate_opponent, p_one probability of choosing 1
  rbinom(trials, 1, p_one)
}

# Single-agent exponential forgetting learner
# input:
  #opponent: the opp actual choice across trials
  #alpha: Learning rate
  #Beta: sensitivity to memory
  #bias: Side bias
  #m0: initial memory
#we will later estimate these parameters with Stan
simulate_memory_agent <- function(opponent, alpha, beta, bias, m0) {
  T <- length(opponent)
  
  memory_before <- numeric(T) # empty vectors to store values
  memory_after  <- numeric(T)
  p_choice      <- numeric(T)
  Self          <- integer(T)
  Feedback      <- integer(T)
  
  m_now <- m0 #initalize current memory state
  
  # loop through trails 1 to T
  for (t in 1:T) {
    memory_before[t] <- m_now # before making the choice, store current memory 
    
    eta_t <- bias + beta * (2 * (m_now - 0.5)) # log odds of choosing 1. Centers memory around 0.5, memory is either leaning towards 0 or 1
    p_choice[t] <- plogis(eta_t) #convert log odds into probability 
    
    Self[t] <- rbinom(1, 1, p_choice[t]) #make stochastic choice, Bernoulli draw, probability of 1 p_choice[t]
    Feedback[t] <- as.integer(Self[t] == opponent[t]) #compute feedback, did the agent match the opponent
    
    # update memory after observing opponent
    m_now <- m_now + alpha * (opponent[t] - m_now) #new memory = old memory + learning rate * prediction error (RW)
    memory_after[t] <- m_now #updated memory after trial t
  }
  
  tibble(
    trial = 1:T,
    Self = Self,
    Other = opponent,
    Feedback = Feedback,
    memory_before = memory_before, #memory state before observing the current outcome
    memory_after = memory_after, #memory state AFTER update
    p_choice = p_choice
  )
}

# Simulation
T <- 100 # 100 trials
opponent <- simulate_opponent(T, p_one = 0.6) #generate 100 opponent choices with prob 0.6 of choosing 1

# simulate one agent playing against the opponent
df_agent <- simulate_memory_agent(
  opponent = opponent,
  alpha = 0.2,
  beta = 2.0,
  bias = 0.0, # no default side bias here
  m0 = 0.5
)

# how memory changes over time - if opponent favors 1, memory should gradually drift upwards
ggplot(df_agent, aes(trial, memory_after)) +
  geom_line() +
  theme_classic()

# Cumulative success rate
ggplot(df_agent, aes(trial, cumsum(Feedback) / trial)) +
  geom_line() +
  theme_classic()



###############################

# --- PREPARE DATA FOR STAN ---

stan_data <- list(
  T = nrow(df_agent),
  Self = as.integer(df_agent$Self), # the agents observed choices 
  Other = as.integer(df_agent$Other) # the opponents observed choices
  # only give Stan the observable values !!!!! -> no memory or p_choice
)



# --- FIT MODEL ---

model <- cmdstan_model("stan_matching_pen.stan")

fit <- model$sample(
  data = stan_data,
  seed = 1999,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

fit$summary(c("alpha", "beta", "bias", "m0"))



###### VALIDATION ############
# --- PRIOR PREDICTIVE CHECK ---

prior_sim <- NULL

for(i in 1:20){ #repeat 20, simulate "fake" df from the priors
  
  alpha <- rbeta(1,2,2)
  beta  <- rlnorm(1,0,0.5)
  bias  <- rnorm(1,0,1)
  m0    <- rbeta(1,2,2)
  
  #generate opponent 
  opponent <- simulate_opponent(100, 0.6)
  
  # Simulate one full "fake" df to using prior drawn parameters
  df <- simulate_memory_agent(opponent, alpha, beta, bias, m0)
  df$sim <- i
  
  prior_sim <- rbind(prior_sim, df) # append df to object defined above 
}

#compute cumulative success rate over time for each simulated data set
prior_sim <- prior_sim %>%
  group_by(sim) %>%
  mutate(cumulative = cumsum(Feedback)/trial)

ggplot(prior_sim, aes(trial, cumulative, group = sim)) +
  geom_line(alpha = 0.3) +
  theme_classic()


# --- POSTERIOR PREDICTIVE ---
#can the fitted model generate data resembling the observed data

posterior <- fit$draws(c("alpha","beta","bias","m0"), format = "df") #extract posterior draws

posterior_sim <- NULL

draws <- sample(1:nrow(posterior), 20)

#loop over sampled posterior rows
for(i in draws){
  
  df <- simulate_memory_agent(
    opponent = df_agent$Other,
    alpha = posterior$alpha[i],
    beta  = posterior$beta[i],
    bias  = posterior$bias[i],
    m0    = posterior$m0[i]
  )
  
  df$draw <- i #label by draw number 
  posterior_sim <- rbind(posterior_sim, df) #append
}

# cumulative success rate over time for each posterior replicated data set
posterior_sim <- posterior_sim %>%
  group_by(draw) %>%
  mutate(cumulative = cumsum(Feedback)/trial)

# thick line: actual observed data
# thin line: posterior simulated data sets 
ggplot(posterior_sim, aes(trial, cumulative, group = draw)) +
  geom_line(alpha = 0.3) +
  geom_line(
    data = df_agent %>% mutate(cumulative = cumsum(Feedback)/trial),
    aes(trial, cumulative),
    inherit.aes = FALSE,   
    linewidth = 1.2
  ) +
  theme_classic()


# --- PRIOR VS POSTERIOR ---

prior_alpha <- rbeta(1000,2,2) #1000 draws from the prior of alpha
post_alpha  <- posterior$alpha # use posterior draws of alpha from the fitted model 

df_plot <- rbind(
  data.frame(value = prior_alpha, type = "Prior"),
  data.frame(value = post_alpha, type = "Posterior")
)

#shows whether the data updated our beliefs about alpha
ggplot(df_plot, aes(value, fill = type)) +
  geom_density(alpha = 0.4) +
  theme_classic() +
  labs(title = "Alpha: Prior vs Posterior")


# --- PARAMETER RECOVERY: ALPHA ---
# 1) Choose true parameter value
# 2) Simulate data from it
# 3) Fit the model back to the data
# 4) See whether the estimates come back near true

alpha_vals <- c(0.1, 0.2, 0.4, 0.6) # test true alpha values

# Hold the other parameters constant (we want to isolate alpha)
beta_fixed <- 2
bias_fixed <- 0
m0_fixed   <- 0.5

n_reps <- 5 # do 5 replications 

results <- NULL

# Loop through true alpha
for(a in alpha_vals){
  
  for(rep in 1:n_reps){
    
    opponent <- simulate_opponent(100, 0.6)
    
    # Generate "fake" data using the known true parameter
    df <- simulate_memory_agent(
      opponent,
      alpha = a,
      beta  = beta_fixed,
      bias  = bias_fixed,
      m0    = m0_fixed
    )
    
    #prepare data for stan
    stan_data <- list(
      T = nrow(df),
      Self = as.integer(df$Self),
      Other = as.integer(df$Other)
    )
    
    # Fit the model back to the simulated data
    fit_rec <- model$sample(
      data = stan_data,
      chains = 4,
      parallel_chains = 4,
      iter_warmup = 1000,
      iter_sampling = 1000,
      seed = 1000 + rep,
      refresh = 0
    )
    
    #Get posterior summary for alpha
    sum <- fit_rec$summary("alpha")
    
    # Store: true alpha, estimated mean alpha, rep number
    results <- rbind(results, data.frame(
      true_alpha = a,
      est_alpha  = sum$mean,
      rep = rep
    ))
  }
}


#plot:
  # Each point is one fitting/simulation run
  # dashed line: perfect recovery
  # diamond: mean estimate at each true value

ggplot(results, aes(true_alpha, est_alpha)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 4, shape = 18) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal(xlim = c(0, 0.7), ylim = c(0, 0.7)) +
  theme_classic() +
  labs(
    title = "Parameter recovery for alpha",
    x = "True alpha",
    y = "Estimated alpha"
  )



# --- PARAMETER RECOVERY: BETA ---

beta_vals <- c(1, 2, 3, 4) 

alpha_fixed <- 0.2
bias_fixed  <- 0
m0_fixed    <- 0.5

n_reps <- 5

results_beta <- NULL

for (b in beta_vals) {
  
  for (rep in 1:n_reps) {
    
    opponent <- simulate_opponent(100, 0.6)
    
    df <- simulate_memory_agent(
      opponent,
      alpha = alpha_fixed,
      beta  = b,
      bias  = bias_fixed,
      m0    = m0_fixed
    )
    
    stan_data <- list(
      T = nrow(df),
      Self = as.integer(df$Self),
      Other = as.integer(df$Other)
    )
    
    fit_rec <- model$sample(
      data = stan_data,
      chains = 4,
      parallel_chains = 4,
      iter_warmup = 1000,
      iter_sampling = 1000,
      seed = 2000 + rep,
      refresh = 0
    )
    
    sum <- fit_rec$summary("beta")
    
    results_beta <- rbind(results_beta, data.frame(
      true_beta = b,
      est_beta  = sum$mean,
      rep = rep
    ))
  }
}

ggplot(results_beta, aes(true_beta, est_beta)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 4, shape = 18) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal(xlim = c(0, 4.5), ylim = c(0, 4.5)) +
  theme_classic() +
  labs(
    title = "Parameter recovery for beta",
    x = "True beta",
    y = "Estimated beta"
  )




# --- PARAMETER RECOVERY: BIAS ---

bias_vals <- c(-1, -0.5, 0, 0.5, 1)

alpha_fixed <- 0.2
beta_fixed  <- 2
m0_fixed    <- 0.5

n_reps <- 5

results_bias <- NULL

for (b in bias_vals) {
  
  for (rep in 1:n_reps) {
    
    opponent <- simulate_opponent(100, 0.6)
    
    df <- simulate_memory_agent(
      opponent,
      alpha = alpha_fixed,
      beta  = beta_fixed,
      bias  = b,
      m0    = m0_fixed
    )
    
    stan_data <- list(
      T = nrow(df),
      Self = as.integer(df$Self),
      Other = as.integer(df$Other)
    )
    
    fit_rec <- model$sample(
      data = stan_data,
      chains = 4,
      parallel_chains = 4,
      iter_warmup = 1000,
      iter_sampling = 1000,
      seed = 3000 + rep,
      refresh = 0
    )
    
    sum <- fit_rec$summary("bias")
    
    results_bias <- rbind(results_bias, data.frame(
      true_bias = b,
      est_bias  = sum$mean,
      rep = rep
    ))
  }
}


ggplot(results_bias, aes(true_bias, est_bias)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 4, shape = 18) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal(xlim = c(-1.2, 1.2), ylim = c(-1.2, 1.2)) +
  theme_classic() +
  labs(
    title = "Parameter recovery for bias",
    x = "True bias",
    y = "Estimated bias"
  )



# --- PARAMETER RECOVERY: M0 ---

m0_vals <- c(0.2, 0.4, 0.6, 0.8)

alpha_fixed <- 0.2
beta_fixed  <- 2
bias_fixed  <- 0

n_reps <- 5

results_m0 <- NULL

for (m in m0_vals) {
  
  for (rep in 1:n_reps) {
    
    opponent <- simulate_opponent(100, 0.6)
    
    df <- simulate_memory_agent(
      opponent,
      alpha = alpha_fixed,
      beta  = beta_fixed,
      bias  = bias_fixed,
      m0    = m
    )
    
    stan_data <- list(
      T = nrow(df),
      Self = as.integer(df$Self),
      Other = as.integer(df$Other)
    )
    
    fit_rec <- model$sample(
      data = stan_data,
      chains = 4,
      parallel_chains = 4,
      iter_warmup = 1000,
      iter_sampling = 1000,
      seed = 4000 + rep,
      refresh = 0
    )
    
    sum <- fit_rec$summary("m0")
    
    results_m0 <- rbind(results_m0, data.frame(
      true_m0 = m,
      est_m0  = sum$mean,
      rep = rep
    ))
  }
}

ggplot(results_m0, aes(true_m0, est_m0)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 4, shape = 18) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_classic() +
  labs(
    title = "Parameter recovery for m0",
    x = "True m0",
    y = "Estimated m0"
  )


# group recovery results intro:
  #average estimated beta
  #spread of estimates

results_beta %>%
  group_by(true_beta) %>%
  summarise(
    mean_est = mean(est_beta),
    sd_est = sd(est_beta)
  )

results_bias %>%
  group_by(true_bias) %>%
  summarise(
    mean_est = mean(est_bias),
    sd_est = sd(est_bias)
  )


results_m0 %>%
  group_by(true_m0) %>%
  summarise(
    mean_est = mean(est_m0),
    sd_est = sd(est_m0)
  )




# --- recovery across trial numbers --- 
trial_values <- c(20, 50, 100, 200)

results_trials <- NULL

#loop through trials
for(n in trial_values){
  
  for(rep in 1:5){
    # simulate opponent with n trials
    opponent <- simulate_opponent(n, 0.6)
    
    #generate data with fixed true parameters 
    df <- simulate_memory_agent(
      opponent,
      alpha = 0.2,
      beta  = 2,
      bias  = 0,
      m0    = 0.5
    )
    
    #prepare Stan
    stan_data <- list(
      T = nrow(df),
      Self = as.integer(df$Self),
      Other = as.integer(df$Other)
    )
    
    # fit (default chain/iterations)
    fit_rec <- model$sample(data = stan_data, refresh = 0)
    
    #only track alpha recov
    sum <- fit_rec$summary("alpha")
    
    #save number of trials and estimated alpha 
    results_trials <- rbind(results_trials, data.frame(
      trials = n,
      est_alpha = sum$mean
    ))
  }
}

#see whether alpha recovery changes over trials
ggplot(results_trials, aes(trials, est_alpha)) +
  geom_point() +
  stat_summary(fun = mean, geom = "point", size = 4) +
  theme_classic()

