data {
  int<lower=1> T;
  array[T] int<lower=0, upper=1> Self;
  array[T] int<lower=0, upper=1> Other;
}

parameters {
  real<lower=0, upper=1> alpha;   // learning rate
  real<lower=0> beta;             // sensitivity to memory
  real bias;                      // side bias
  real<lower=0, upper=1> m0;      // initial memory
}

transformed parameters {
  vector[T] eta;
  vector[T] p;
  vector[T] memory_before;
  vector[T] memory_after;

  real m_now;
  m_now = m0;

  for (t in 1:T) {
    memory_before[t] = m_now;

    eta[t] = bias + beta * (2 * (m_now - 0.5));
    p[t] = inv_logit(eta[t]);

    m_now = m_now + alpha * (Other[t] - m_now);
    memory_after[t] = m_now;
  }
}

model {
  alpha ~ beta(2, 2);
  beta ~ lognormal(0, 0.5);
  bias ~ normal(0, 1);
  m0 ~ beta(2, 2);

  for (t in 1:T) {
    Self[t] ~ bernoulli_logit(eta[t]);
  }
}

generated quantities {
  array[T] int y_rep;
  vector[T] log_lik;

  for (t in 1:T) {
    y_rep[t] = bernoulli_logit_rng(eta[t]);
    log_lik[t] = bernoulli_logit_lpmf(Self[t] | eta[t]);
  }
}

