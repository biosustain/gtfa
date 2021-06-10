functions {
#include custom_functions.stan
}
data {
  // network properties
  int<lower=1> N_metabolite;
  int<lower=1> N_drain;
  int<lower=1> N_enzyme;
  int<lower=1> N_reaction;
  int<lower=1> N_b_free;
  int<lower=1> N_b_bound;
  matrix[N_metabolite, N_reaction] S;
  array[N_b_bound] int<lower=1,upper=N_enzyme> ix_b_bound;
  array[N_b_free] int<lower=1,upper=N_enzyme> ix_b_free;
  array[N_reaction] int<lower=0,upper=N_enzyme> reaction_to_enzyme;  // zero if no enzyme, otherwise index
  array[N_reaction] int<lower=0,upper=N_drain> reaction_to_drain;    // zero if no drain, otherwise index
  // measurements
  int<lower=1> N_condition;
  int<lower=1> N_y_enzyme;
  int<lower=1> N_y_metabolite;
  int<lower=1> N_y_flux;
  vector<lower=0>[N_y_enzyme] y_enzyme;
  vector<lower=0>[N_y_enzyme] sigma_enzyme;
  array[N_y_enzyme] int<lower=1,upper=N_enzyme> enzyme_y_enzyme;
  array[N_y_enzyme] int<lower=1,upper=N_condition> condition_y_enzyme;
  vector<lower=0>[N_y_metabolite] y_metabolite;
  vector<lower=0>[N_y_metabolite] sigma_metabolite;
  array[N_y_metabolite] int<lower=1,upper=N_metabolite> metabolite_y_metabolite;
  array[N_y_metabolite] int<lower=1,upper=N_condition> condition_y_metabolite;
  vector[N_y_flux] y_flux;
  vector<lower=0>[N_y_flux] sigma_flux;
  array[N_y_flux] int<lower=1,upper=N_reaction> reaction_y_flux;
  array[N_y_flux] int<lower=1,upper=N_condition> condition_y_flux;
  // priors
  array[2] vector[N_metabolite] prior_dgf; //independent normal
  array[2, N_condition] vector[N_drain] prior_drain;
  array[2, N_condition] vector[N_b_free] prior_b_free;
  array[2, N_condition] vector[N_enzyme] prior_enzyme;
  array[2, N_condition] vector[N_metabolite] prior_metabolite;
  // config
  array[N_condition] vector[N_b_bound] b_bound_guess;
  int<lower=0,upper=1> likelihood;
  real rel_tol;
  real function_tol;
  int max_num_steps;
}
transformed data {
  array[rows(S)*cols(S)] real x_r = to_array_1d(S);
  array[4 + 2 * N_enzyme] int x_i =
    append_array({N_metabolite, N_enzyme, N_reaction, N_b_free, N_b_bound},
                 append_array(reaction_to_enzyme, append_array(reaction_to_drain, append_array(ix_b_free, ix_b_bound))));
}
parameters {
  vector[N_metabolite] dgf_z;
  array[N_condition] vector<lower=0>[N_b_free] b_free;
  array[N_condition] vector<lower=0>[N_drain] drain;
  array[N_condition] vector<lower=0>[N_enzyme] enzyme;
  array[N_condition] vector<lower=0>[N_metabolite] metabolite;
}
transformed parameters {
  vector[N_metabolite] dgf = prior_dgf[1] + dgf_z .* prior_dgf[2];
  array[N_condition] vector[N_reaction] flux;
  array[N_condition] vector[N_b_bound] b_bound;
  for (c in 1:N_condition){
    int N_theta = rows(dgf) + rows(b_free[c]) + rows(drain[c]) + rows(enzyme[c]) + rows(metabolite[c]);
    vector[N_theta] theta = get_theta(dgf, b_free[c], drain[c], enzyme[c], metabolite[c]);
    // Function, initial guess, control params
    // Theta - vector of parameters
    // x_r - array of real valued data variables. S.
    // x_i - array of integer valued data variables
    //
    b_bound[c] = algebra_solver_newton(steady_state, b_bound_guess[c], theta, x_r, x_i, rel_tol, function_tol, max_num_steps);
    flux[c] = get_flux(S, b_bound[c], theta, x_i);
  }
}
model {
  dgf_z ~ normal(0, 1);
  for (c in 1:N_condition){
    drain[c] ~ lognormal(prior_drain[1, c], prior_drain[2, c]);
    enzyme[c] ~ lognormal(prior_enzyme[1, c], prior_enzyme[2, c]);
    metabolite[c] ~ lognormal(prior_metabolite[1, c], prior_metabolite[2, c]);
    b_free[c] ~ lognormal(prior_b_free[1, c], prior_b_free[2, c]);
  }
  if (likelihood == 1){
    for (n in 1:N_y_enzyme){
      int c = condition_y_enzyme[n];
      int e = enzyme_y_enzyme[n];
      y_enzyme[n] ~ lognormal(enzyme[c, e], sigma_enzyme[n]);
    }
    for (n in 1:N_y_metabolite){
      int c = condition_y_metabolite[n];
      int m = metabolite_y_metabolite[n];
      y_metabolite[n] ~ lognormal(metabolite[c, m], sigma_metabolite[n]);
    }
    for (n in 1:N_y_flux){
      int c = condition_y_flux[n];
      int r = reaction_y_flux[n];
      y_flux[n] ~ lognormal(flux[c, r], sigma_flux[n]);
    }
  }
}
generated quantities {
  array[N_condition] vector[N_enzyme] dgr;
  for (c in 1:N_condition) dgr[c] = get_dgr(S, dgf, metabolite[c]);
}

