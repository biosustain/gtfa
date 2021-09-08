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
  int<lower=1> N_reaction;
  int<lower=1> N_transport;
  int<lower=1> N_enzyme;

  int<lower=0> N_fixed_transport;
  int<lower=0> N_free_transport;

  int<lower=0> N_fixed_met_conc;
  int<lower=0> N_free_met_conc;

  int<lower=1> N_free_x;
  int<lower=0> N_fixed_x;
  int<lower=1> N_x;

  matrix[N_metabolite, N_reaction] S;

  // Indexing - Some of these are not used but could be convenient later
  array[N_free_met_conc] int<lower=1, upper=N_metabolite> ix_free_to_met;
  array[N_fixed_met_conc] int<lower=1, upper=N_metabolite> ix_fixed_to_met;
  //
  array[N_free_transport] int<lower=1, upper=N_reaction> ix_free_to_trans;
  array[N_fixed_transport] int<lower=1, upper=N_reaction> ix_fixed_to_trans;
  //
  array[N_enzyme] int<lower=1, upper=N_reaction> ix_enzyme;
  array[N_transport] int<lower=1, upper=N_reaction> ix_transport;
  // Transport fixed and free reactions
  array[N_free_x] int<lower=1,upper=N_x> ix_free;
  array[N_fixed_x] int<lower=1,upper=N_x> ix_fixed;
  // Transport fixed and free reactions
  array[N_fixed_transport] int<lower=1,upper=N_reaction> ix_fixed_transport;
  array[N_free_transport] int<lower=1,upper=N_reaction> ix_free_transport;
  // Concentration fixed and free
  array[N_fixed_met_conc] int<lower=1,upper=N_reaction> ix_fixed_met_conc;
  array[N_free_met_conc] int<lower=1,upper=N_reaction> ix_free_met_conc;
  // // measurements
  int<lower=1> N_condition;
  int<lower=0> N_y_metabolite;
  int<lower=0> N_y_flux;
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
  array[2, N_condition] vector[N_enzyme] prior_enzyme;
  array[2, N_condition] vector[N_enzyme] prior_b;
  array[2, N_condition] vector[N_free_met_conc] prior_free_met_conc;
  // config
  int<lower=0,upper=1> likelihood;
}
transformed data {
    // The combined matrix for transport reactions and concentrations
    matrix[N_reaction, N_metabolite + N_transport] s_gamma = rep_matrix(0, N_reaction, N_metabolite + N_transport);
    vector[N_transport] diag_vals = rep_vector(1, N_transport);
    s_gamma[:N_transport, :N_transport] = diag_matrix(diag_vals);
    s_gamma[N_transport+1:, N_transport+1:] = S'[ix_enzyme];
}

parameters {
  vector[N_metabolite] dgf;
  array[N_condition] vector<lower=0>[N_enzyme] b;
  array[N_condition] vector<lower=0>[N_enzyme] enzyme;
  array[N_condition] vector[N_free_met_conc] log_metabolite_free;
  array[N_condition] vector[N_free_transport] transport_free;
}

transformed parameters {
  array[N_condition] vector[N_reaction] dgr;
  array[N_condition] vector[N_reaction] flux;
  array[N_condition] vector[N_metabolite] log_metabolite;
  real RT = 0.008314 * 298.15;
  //
  for (cond in 1:N_condition){
    vector[N_metabolite] rhs;
    matrix[N_metabolite, N_reaction] mod_s;
    matrix[N_metabolite, N_metabolite + N_transport] s_total;
    vector[N_free_x] free_x;
    vector[N_fixed_x] fixed_x;
    // Determine the final matrix to be solved
    mod_s = S;
    // This could be done with the left/right multiply function maybe?
    mod_s[:, ix_enzyme] = mod_s[:, ix_enzyme] .* rep_matrix((b[cond] .* enzyme[cond])', N_metabolite);
    s_total = mod_s * s_gamma;
    // Get a vector of the free values
    if (N_free_transport > 0){
        free_x[:N_free_transport] = transport_free[cond];
    }
    free_x[N_free_transport+1:] = dgf[ix_free_met_conc] + RT * log_metabolite_free[cond];
    rhs = -s_total[:, ix_free] * free_x;
    fixed_x = s_total[:, ix_fixed] \ rhs;
    // Solve for the remaining log metabolite values
    log_metabolite[cond, ix_free_met_conc] = log_metabolite_free[cond];
    log_metabolite[cond, ix_fixed_met_conc] = (fixed_x[ix_fixed_to_met] - dgf[ix_fixed_met_conc]) ./ RT;
    // print("Calc log met");
    // print(fixed_x[ix_fixed_to_met]);
    // print(dgf[ix_fixed_met_conc]);
    // print((fixed_x[ix_fixed_to_met] - dgf[ix_fixed_met_conc]) ./ RT);
    // Calculate the dgr
    dgr[cond] = get_dgr(dgf, log_metabolite[cond], S);
    // Calculate the fluxes
    flux[cond][ix_enzyme] = dgr[cond][ix_enzyme] .* b[cond] .* enzyme[cond];
    // Add the fixed and free transport reactions
    flux[cond][ix_free_transport] = transport_free[cond];
    flux[cond][ix_fixed_transport] = fixed_x[ix_fixed_to_trans];
    // print("b");
    // print(b[cond]);
    // print("Enz");
    // print(enzyme[cond]);
    // print("fixed");
    // print(fixed_x);
    // print("Free");
    // print(free_x);
    // print("Log met free");
    // print(log_metabolite_free);
    // print("dgf");
    // print(dgf);
    // print("log");
    // print(log_metabolite[cond]);
    // print("dgr");
    // print(dgr[cond]);
    // print("flux");
    // print(flux[cond]);
  }
}
model {
  dgf ~ normal(prior_dgf[1], prior_dgf[2]);
  for (c in 1:N_condition){
    enzyme[c] ~ lognormal(prior_enzyme[1, c], prior_enzyme[2, c]);
    b[c] ~ lognormal(prior_b[1, c], prior_b[2, c]);
    log_metabolite_free[c] ~ normal(prior_free_met_conc[1, c], prior_free_met_conc[2, c]);
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

generated quantities {
  // Check that the fluxes follow steady-state
  {
    vector[N_metabolite] conc_change;
    real eps = 1e-6;
    // Should be all 0
    for (cond in 1:N_condition){
      conc_change = S * flux[cond];
      print(conc_change);
      for (i in 1:N_metabolite){
        if (fabs(conc_change[i]) > eps){
          reject("The steady state assumption must hold. Met ", i,
          "concentration changed ", conc_change[i]);
        }
      }
    }
  }
}
