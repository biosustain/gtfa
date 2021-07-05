functions {
  vector get_dgr(vector dgf, vector log_metabolite, matrix S){
    /* Get delta G of reaction for each reaction. */
    real RT = 0.008314 * 298.15;
    return S' * (dgf + RT * log_metabolite);
  }
}
data {
  // network properties
  int<lower=1> N_metabolite;
  int<lower=1> N_transport;
  int<lower=1> N_enzyme;
  int<lower=1> N_reaction;
  int<lower=1> N_free_flux;
  int<lower=0> N_free_enzyme;
  int<lower=0> N_free_transport;
  matrix[N_metabolite, N_reaction] S;
  matrix[N_reaction, N_free_flux] NS;
  array[N_free_flux] int<lower=1,upper=N_reaction> ix_free_flux;  // what is the reaction index of each free flux?
  array[N_free_enzyme] int<lower=1,upper=N_enzyme> ix_free_enzyme;         // what is the free flux for each free enzyme?
  array[N_free_transport] int<lower=1,upper=N_transport> ix_free_transport;   // what is the free flux for each free transport?
  // measurements
  int<lower=1> N_condition;
  int<lower=1> N_y_metabolite;
  int<lower=1> N_y_flux;
  vector<lower=0>[N_y_metabolite] y_metabolite;
  vector<lower=0>[N_y_metabolite] sigma_metabolite;
  array[N_y_metabolite] int<lower=1,upper=N_metabolite> metabolite_y_metabolite;
  array[N_y_metabolite] int<lower=1,upper=N_condition> condition_y_metabolite;
  vector[N_y_flux] y_flux;
  vector<lower=0>[N_y_flux] sigma_flux;
  array[N_y_flux] int<lower=1,upper=N_reaction> reaction_y_flux;
  array[N_y_flux] int<lower=1,upper=N_condition> condition_y_flux;
  // priors
  array[2] vector[N_metabolite] prior_dgf;
  array[2, N_condition] vector[N_free_transport] prior_transport_free;
  array[2, N_condition] vector[N_free_enzyme] prior_enzyme_free;
  array[2, N_condition] vector[N_free_enzyme] prior_b_free;
  array[2, N_condition] vector[N_metabolite] prior_log_metabolite;
  // config
  int<lower=0,upper=1> likelihood;
}
parameters {
  vector<offset=prior_dgf[1], multiplier=prior_dgf[2]>[N_metabolite] dgf;
  vector<lower=0>[N_free_enzyme] b_free[N_condition];
  vector<lower=0>[N_free_enzyme] enzyme_free[N_condition];
  vector[N_metabolite] log_metabolite[N_condition];
  vector[N_free_transport] transport_free[N_condition];
}
transformed parameters {
  array[N_condition] vector[N_reaction] dgr;
  array[N_condition] vector[N_reaction] flux;
  for (c in 1:N_condition){
    dgr[c] = get_dgr(dgf, log_metabolite[c], S);
    vector[N_free_flux] free_flux_c;
    free_flux_c[ix_free_enzyme] =
      -dgr[c, ix_free_flux[ix_free_enzyme]] .* b_free[c] .* enzyme_free[c];
    free_flux_c[ix_free_transport] = transport_free[c];
    flux[c] = NS * free_flux_c;
  }
}
model {
  dgf ~ normal(prior_dgf[1], prior_dgf[2]);
  for (c in 1:N_condition){
    enzyme_free[c] ~ lognormal(prior_enzyme_free[1, c], prior_enzyme_free[2, c]);
    b_free[c] ~ lognormal(prior_b_free[1, c], prior_b_free[2, c]);
    log_metabolite[c] ~ normal(prior_log_metabolite[1, c], prior_log_metabolite[2, c]);
    transport_free[c] ~ normal(prior_transport_free[1, c], prior_transport_free[2, c]);
  }
  if (likelihood == 1){
    for (n in 1:N_y_metabolite){
      int c = condition_y_metabolite[n];
      int m = metabolite_y_metabolite[n];
      y_metabolite[n] ~ lognormal(log_metabolite[c, m], sigma_metabolite[n]);
    }
    for (n in 1:N_y_flux){
      int c = condition_y_flux[n];
      int r = reaction_y_flux[n];
      y_flux[n] ~ normal(flux[c, r], sigma_flux[n]);
    }
  }
}

