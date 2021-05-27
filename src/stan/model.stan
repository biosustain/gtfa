functions {
#include custom_functions.stan
}
data {
  // network properties
  int<lower=1> N_metabolite;
  int<lower=1> N_drain;
  int<lower=1> N_reaction;
  int<lower=1> N_b_free;
  matrix[N_metabolite, N_reaction] S;
  array[N_reaction] int<lower=0,upper=1> is_drain;
  int<lower=1,upper=N_enzyme> ix_b_bound[N_reaction-N_drain-N_b_free];
  int<lower=1,upper=N_enzyme> ix_b_free[N_b_free];
  array[N_enzyme] int<lower=1,upper=N_reaction> enzyme_to_reaction;
  // measurements
  int<lower=1> N_condition;
  int<lower=1> N_y_enzyme;
  int<lower=1> N_y_metabolite;
  int<lower=1> N_y_flux;
  array[N_y_enzyme] int<lower=1,upper=N_enzyme> enzyme_y_enzyme;
  array[N_y_metabolite] int<lower=1,upper=N_metabolite> metabolite_y_metabolite;
  array[N_y_flux] int<lower=1,upper=N_reaction> reaction_y_flux;
  array[N_y_enzyme] int<lower=1,upper=N_condition> condition_y_enzyme;
  array[N_y_metabolite] int<lower=1,upper=N_condition> condition_y_metabolite;
  array[N_y_flux] int<lower=1,upper=N_condition> condition_y_flux;
  vector<lower=0>[N_y_enzyme] y_enzyme;
  vector<lower=0>[N_y_metabolite] y_metabolite;
  vector[N_y_flux] y_flux;
  real<lower=0> sigma_flux;
  real<lower=0> sigma_metabolite;
  real<lower=0> sigma_enzyme;
  // priors
  array[2] vector[N_metabolite] prior_dgf;
  array[2, N_condition] vector[N_b_free] prior_b_free;
  array[2, N_condition] vector[N_enzyme] prior_enzyme;
  array[2, N_condition] vector[N_metabolite] prior_metabolite;
  // config
  int<lower=0,upper=1> likelihood;
}
transformed data {
  int N_b_bound = N_enzyme - N_b_free;
  int N_theta = get_N_theta(N_metabolite, N_b_free, N_enzyme);
  int N_x_i = get_N_x_i(N_enzyme);
  int N_x_r = get_N_x_r(get_x_r(S));
  array[N_x_i] int x_i = get_x_i(N_metabolite,
                                 N_reaction,
                                 enzyme_to_reaction,
                                 ix_b_free,
                                 ix_b_bound);
  array[N_x_r] real x_r = get_x_r(S);
  array[N_condition] vector[N_theta] theta_guess;
}
parameters {
  vector[N_metabolite] dgf;
  array[N_condition] vector<lower=0>[N_b_free] b_free;
  array[N_condition] vector<lower=0>[N_enzyme] enzyme;
  array[N_condition] vector<lower=0>[N_metabolite] metabolite;
}
transformed parameters {
  array[N_condition] vector[N_reaction] flux;
  array[N_condition] vector[N_b_bound] b_bound;

  for (c in 1:N_condition){
    vector[N_theta] theta = get_theta(dgf, b_free[c], enzyme[c], metabolite[c]);
    b_bound[c] = algebra_solver_newton(steady_state, theta_guess[c], theta, x_r, x_i);
    flux[c] = get_flux(S, b_bound[c], theta, x_i);
  }
}
model {
  dgf ~ normal(prior_dgf[1], prior_dgf[2]);
  for (c in 1:N_condition){
    enzyme[c] ~ lognormal(prior_enzyme[1, c], prior_enzyme[2, c]);
    metabolite[c] ~ lognormal(prior_metabolite[1, c], prior_metabolite[2, c]);
    b_free[c] ~ normal(prior_b_free[1, c], prior_b_free[2, c]);
  }
  if (likelihood == 1){
    for (n in 1:N_y_enzyme){
      int c = condition_y_enzyme[n];
      int e = enzyme_y_enzyme[n];
      y_enzyme[n] ~ lognormal(enzyme[c, e], sigma_enzyme);
    }
    for (n in 1:N_y_metabolite){
      int c = condition_y_metabolite[n];
      int m = metabolite_y_metabolite[n];
      y_metabolite[n] ~ lognormal(metabolite[c, m], sigma_metabolite);
    }
    for (n in 1:N_y_flux){
      int c = condition_y_flux[n];
      int r = reaction_y_flux[n];
      y_flux[n] ~ lognormal(flux[c, r], sigma_flux);
    }
  }
}
generated quantities {
  array[N_condition] vector[N_enzyme] dgr;
  for (c in 1:N_condition) dgr[c] = get_dgr(S, dgf, metabolite[c]);
}
