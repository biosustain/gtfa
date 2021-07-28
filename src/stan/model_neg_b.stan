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
  int<lower=0> N_free_transport;
  int<lower=0> N_free_enzyme;
  matrix[N_metabolite, N_reaction] S;
  // Indexing - Some of these are not used but could be convenient later
  // Internal fixed and free reaction
  array[N_enzyme - N_free_enzyme] int<lower=1,upper=N_reaction> ix_fixed_enzyme;
  array[N_free_enzyme] int<lower=1,upper=N_reaction> ix_free_enzyme;
  // Now maps to the enzyme reactions
  array[N_enzyme - N_free_enzyme] int<lower=1,upper=N_enzyme> ix_fixed_to_enzyme;
  array[N_free_enzyme] int<lower=1,upper=N_enzyme> ix_free_to_enzyme;
  // Transport fixed and free reactions
  array[N_transport- N_free_transport] int<lower=1,upper=N_reaction> ix_fixed_transport;
  array[N_free_transport] int<lower=1,upper=N_reaction> ix_free_transport;
  // Now maps to the transport reactions
  array[N_transport - N_free_transport] int<lower=1,upper=N_transport> ix_fixed_to_transport;
  array[N_free_transport] int<lower=1,upper=N_transport> ix_free_to_transport;
  // // All fixed and free
  array[N_reaction - N_free_flux] int<lower=1,upper=N_reaction> ix_fixed_flux;
  array[N_free_flux] int<lower=1,upper=N_reaction> ix_free_flux;
  // A matrix for calculating fixed variables corresponding to 0 = Sv
  matrix[N_reaction - N_free_flux, N_free_flux] free_to_fixed;


  // // measurements
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
  array[2, N_condition] vector[N_enzyme] prior_enzyme;
  array[2, N_condition] vector[N_free_flux] prior_b_free;
  array[2, N_condition] vector[N_metabolite] prior_log_metabolite;
  // config
  int<lower=0,upper=1> likelihood;
}
transformed data {
  // print("Transformed data");
}

parameters {
  vector<offset=prior_dgf[1], multiplier=prior_dgf[2]>[N_metabolite] dgf; // Constant m
  array[N_condition] vector<lower=0>[N_free_enzyme] b_free; // b_free + transport_free = r - rank(S)
  array[N_condition] vector<lower=0>[N_enzyme] enzyme; //
  array[N_condition] vector[N_metabolite] log_metabolite;
  array[N_condition] vector[N_free_transport] transport_free;
}
transformed parameters {
  array[N_condition] vector[N_reaction] dgr;
  array[N_condition] vector[N_reaction] flux;
  array[N_condition] vector[N_enzyme] b;
  for (cond in 1:N_condition){
    // Calculate the dgr of all fluxes
    dgr[cond] = get_dgr(dgf, log_metabolite[cond], S);
    // Calculate the free fluxes with the free b values
    flux[cond, ix_free_enzyme] = -dgr[cond, ix_free_enzyme] .* b_free[cond] .* enzyme[cond, ix_free_to_enzyme];
    // Add the sampled free transport
    flux[cond, ix_free_transport] = transport_free[cond];
    // Calculate and store the fixed fluxes in the flux vector
    flux[cond, ix_fixed_flux] = free_to_fixed * flux[cond, ix_free_flux];
    // Calculate the fixed b values from the fixed fluxes
    b[cond, ix_free_to_enzyme] = b_free[cond];
    b[cond, ix_fixed_to_enzyme] = flux[cond, ix_fixed_enzyme]
    ./ (enzyme[cond, ix_fixed_to_enzyme] .* -dgr[cond, ix_fixed_enzyme]);
  }

}
model {
  dgf ~ normal(prior_dgf[1], prior_dgf[2]);
  for (c in 1:N_condition){
    enzyme[c] ~ lognormal(prior_enzyme[1, c], prior_enzyme[2, c]);
    b_free[c] ~ lognormal(1, prior_b_free[2, c]);
    log_metabolite[c] ~ normal(prior_log_metabolite[1, c], prior_log_metabolite[2, c]);
    transport_free[c] ~ normal(prior_transport_free[1, c], prior_transport_free[2, c]);
  }
  // Do not allow negative b values
  for (c in 1:N_condition){
    for (enz in 1:N_enzyme){
      target += fmin(0.0, b[c, enz] * 1000);
    }
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





  // for (n in 1:N_y_metabolite){
  //   int c = condition_y_metabolite[n];
  //   int m = metabolite_y_metabolite[n];
  //   y_metabolite[n] ~ lognormal(log_metabolite[c, m], sigma_metabolite[n]);
  // }
  // for (n in 1:N_y_flux){
  //   int c = condition_y_flux[n];
  //   int r = reaction_y_flux[n];
  //   y_flux[n] ~ normal(flux[c, r], sigma_flux[n]);
  // }


}

generated quantities {
  // Check that the fluxes follow stead-state
  // Check that all of the fluxes follow the delta-g
  int neg_b = 0;
  for (c in 1:N_condition){
    for (i in 1:N_enzyme){
        if (b[c,i] < 0){
          neg_b = 1;
        }
    }
  }
  {
    vector[N_metabolite] conc_change;
    real eps = 1e-6;
    // Should be all 0
    for (cond in 1:N_condition){
      conc_change = S * flux[cond];
      for (i in 1:N_metabolite){
        if (fabs(conc_change[i]) > eps){
          reject("The steady state assumption must hold. Met ", i,
          "concentration changed ", conc_change[i]);
        }
      }
    }
  }
}
