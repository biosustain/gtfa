functions {
  vector get_dgr(vector dgf, vector log_metabolite, matrix s_intern){
    /* Get delta G of reaction for each reaction. */
    real RT = 0.008314 * 298.15;
    return s_intern' * (dgf + RT * log_metabolite);
  }
}
data {
  //// network properties
  int<lower=1> N_metabolite;
  int<lower=1> N_reaction;
  int<lower=1> N_exchange;
  int<lower=1> N_internal;
  //
  int<lower=0> N_fixed_exchange;
  int<lower=0> N_free_exchange;
  //
  int<lower=0> N_fixed_met_conc;
  int<lower=0> N_free_met_conc;
  //
  int<lower=1> N_free_x;
  int<lower=0> N_fixed_x;
  int<lower=1> N_x;
  //
  int<lower=1> N_free_rows;
  int<lower=0> N_fixed_rows;
  //
  matrix[N_metabolite, N_reaction] S;
  //// Indexing - Some of these are not used but could be convenient later
  // Which elements of the free x are metabolites and which are exchange rxns
  array[N_free_met_conc] int<lower=1, upper=N_free_x> ix_free_met_to_free;
  array[N_free_exchange] int<lower=1, upper=N_free_x> ix_free_ex_to_free;
  // Which elements of the fixed x are metabolites and which are exchange rxns
  array[N_fixed_met_conc] int<lower=1, upper=N_fixed_x> ix_fixed_met_to_fixed;
  array[N_fixed_exchange] int<lower=1, upper=N_fixed_x> ix_fixed_ex_to_fixed;
  // Which elemets of x are free and fixed
  array[N_free_x] int<lower=1, upper=N_x> ix_free_to_x;
  array[N_fixed_x] int<lower=1, upper=N_x> ix_fixed_to_x;
  // Which elements of x are exchange reactions and which are metabolites
  array[N_exchange] int<lower=1, upper=N_x> ix_ex_to_x;
  array[N_metabolite] int<lower=1, upper=N_x> ix_met_to_x;
  // Which reactions are internal and exchange
  array[N_internal] int<lower=1, upper=N_reaction> ix_internal_to_rxn;
  array[N_exchange] int<lower=1, upper=N_reaction> ix_ex_to_rxn;
  // Which metabolites are free and fixed
  array[N_free_met_conc] int<lower=1, upper=N_metabolite> ix_free_met_to_met;
  array[N_fixed_met_conc] int<lower=1, upper=N_metabolite> ix_fixed_met_to_met;
  // Which exchange reactions are free and fixed
  array[N_free_exchange] int<lower=1, upper=N_exchange> ix_free_ex_to_ex;
  array[N_fixed_exchange] int<lower=1, upper=N_exchange> ix_fixed_ex_to_ex;
  // Which rows of the S_c matrix are necesary for solving the system of
  // linear equations.
  array[N_free_rows] int<lower=1, upper=N_metabolite> ix_free_row_to_met;
  array[N_fixed_rows] int<lower=1, upper=N_metabolite> ix_fixed_row_to_met;
  //// measurements
  int<lower=1> N_condition;
  int<lower=0> N_y_enzyme;
  int<lower=0> N_y_metabolite;
  int<lower=0> N_y_flux;
  // Fluxes
  vector[N_y_flux] y_flux;
  vector<lower=0>[N_y_flux] sigma_flux;
  array[N_y_flux] int<lower=1,upper=N_reaction> reaction_y_flux;
  array[N_y_flux] int<lower=1,upper=N_condition> condition_y_flux;
  // Enzymes
  vector[N_y_enzyme] y_enzyme;
  vector<lower=0>[N_y_enzyme] sigma_enzyme;
  array[N_y_enzyme] int<lower=1,upper=N_internal> internal_y_enzyme;
  array[N_y_enzyme] int<lower=1,upper=N_condition> condition_y_enzyme;
  // Mets
  vector[N_y_metabolite] y_metabolite;
  vector<lower=0>[N_y_metabolite] sigma_metabolite;
  array[N_y_metabolite] int<lower=1,upper=N_metabolite> metabolite_y_metabolite;
  array[N_y_metabolite] int<lower=1,upper=N_condition> condition_y_metabolite;

  //// priors
  vector[N_metabolite] prior_dgf_mean;
  matrix[N_metabolite, N_metabolite] prior_dgf_cov;
  array[2, N_condition] vector[N_free_exchange] prior_exchange_free;
  array[2, N_condition] vector[N_internal] prior_enzyme;
  array[2, N_condition] vector[N_internal] prior_b;
  array[2, N_condition] vector[N_free_met_conc] prior_free_met_conc;
}
transformed data {
    // The combined matrix for exchange reactions and concentrations
    matrix[N_reaction, N_metabolite + N_exchange] s_gamma = rep_matrix(0, N_reaction, N_metabolite + N_exchange);
    vector[N_exchange] ones = rep_vector(1, N_exchange);
    s_gamma[ix_ex_to_rxn, ix_ex_to_x] = diag_matrix(ones);
    // The matrix is inverted as well because fluxes are the opposite direction to the dgr (-dgr means positive fluxes)
    s_gamma[ix_internal_to_rxn, ix_met_to_x] = -S'[ix_internal_to_rxn];
    //// Some extra indices for convenience
    // Directly from free and fixed metabolites and exchange rxns to the x vector
    array[N_free_met_conc] int<lower=1, upper=N_x> ix_free_met_to_x = ix_free_to_x[ix_free_met_to_free];
    array[N_fixed_met_conc] int<lower=1, upper=N_x> ix_fixed_met_to_x = ix_fixed_to_x[ix_fixed_met_to_fixed];
    array[N_free_exchange] int<lower=1, upper=N_x> ix_free_ex_to_x = ix_free_to_x[ix_free_ex_to_free];
    array[N_fixed_exchange] int<lower=1, upper=N_x> ix_fixed_ex_to_x = ix_free_to_x[ix_fixed_ex_to_fixed];
    // Free and fixed exchanges to their corresponding reactions
    array[N_free_exchange] int<lower=1, upper=N_reaction> ix_free_met_to_mets = ix_ex_to_rxn[ix_free_ex_to_ex];
    array[N_fixed_exchange] int<lower=1, upper=N_reaction> ix_fixed_met_to_mets = ix_ex_to_rxn[ix_fixed_ex_to_ex];
}

parameters {
  vector[N_metabolite] dgf_ctd;
  array[N_condition] vector[N_internal] b;
  array[N_condition] vector[N_internal] log_enzyme;
  array[N_condition] vector[N_free_met_conc] log_metabolite_free;
  array[N_condition] vector[N_free_exchange] exchange_free;
}

transformed parameters {
  array[N_condition] vector[N_internal] dgr;
  array[N_condition] vector[N_reaction] flux;
  array[N_condition] vector[N_metabolite] log_metabolite;
  array[N_condition] vector[N_x] x;
  // Recenter the dgf
  vector[N_metabolite] dgf = dgf_ctd + prior_dgf_mean;
  //
  for (cond in 1:N_condition){
    real RT = 0.008314 * 298.15;
    vector[N_metabolite] rhs;
    matrix[N_metabolite, N_reaction] mod_s;
    matrix[N_metabolite, N_metabolite + N_exchange] s_c;
    // Determine the final matrix to be solved
    mod_s = S;
    // This could be done with the left/right multiply function maybe?
    mod_s[:, ix_internal_to_rxn] = mod_s[:, ix_internal_to_rxn] .*
      rep_matrix((exp(b[cond]) .* exp(log_enzyme[cond]))', N_metabolite);
    s_c = mod_s * s_gamma;
    // Get a vector of the free values
    if (N_free_exchange > 0){
        x[cond][ix_free_ex_to_x] = exchange_free[cond];
    }
    x[cond][ix_free_met_to_x] = dgf[ix_free_met_to_met] + RT * log_metabolite_free[cond];
    rhs = -s_c[ix_free_row_to_met, ix_free_to_x] * x[cond][ix_free_to_x];
    x[cond][ix_fixed_to_x] = s_c[ix_free_row_to_met, ix_fixed_to_x] \ rhs;
    for (i in ix_fixed_to_x){
      if (is_nan(x[cond][i])){
        reject("The linear equations could not be solved");
      }
    }
    // Solve for the remaining log metabolite values
    log_metabolite[cond, ix_free_met_to_met] = log_metabolite_free[cond];
    log_metabolite[cond, ix_fixed_met_to_met] = (x[cond][ix_fixed_met_to_x] -
        dgf[ix_fixed_met_to_met]) ./ RT;
    // Calculate the dgr
    dgr[cond] = get_dgr(dgf, log_metabolite[cond], S[:, ix_internal_to_rxn]);
    // Calculate the fluxes
    flux[cond][ix_internal_to_rxn] = -dgr[cond] .* exp(b[cond]) .* exp(log_enzyme[cond]);
    // Add the fixed and free exchange reactions
    flux[cond][ix_ex_to_rxn] = x[cond][ix_ex_to_x];
  }
}
model {
  //// Priors
  // Dgf
  dgf_ctd ~ multi_normal(rep_vector(0, N_metabolite), prior_dgf_cov);
  for (c in 1:N_condition){
    b[c] ~ normal(prior_b[1, c], prior_b[2, c]);
    log_enzyme[c] ~ normal(prior_enzyme[1, c], prior_enzyme[2, c]);
    log_metabolite_free[c] ~ normal(prior_free_met_conc[1, c], prior_free_met_conc[2, c]);
    exchange_free[c] ~ normal(prior_exchange_free[1, c], prior_exchange_free[2, c]);
  }
  //// Likelihood
  // Metabolite concentrations
  for (n in 1:N_y_metabolite){
    int c = condition_y_metabolite[n];
    int m = metabolite_y_metabolite[n];
    y_metabolite[n] ~ student_t(4, log_metabolite[c, m], sigma_metabolite[n]);
  }
  // Enzyme concentrations
  for (n in 1:N_y_enzyme){
    int c = condition_y_enzyme[n];
    int enz_ind = internal_y_enzyme[n];
    y_enzyme[n] ~ student_t(4, log_enzyme[c, enz_ind], sigma_enzyme[n]);
  }
  // Fluxes
  for (n in 1:N_y_flux){
    int c = condition_y_flux[n];
    int r = reaction_y_flux[n];
    y_flux[n] ~ student_t(4, flux[c, r], sigma_flux[n]);
  }
}

generated quantities {
  // Check that the fluxes follow steady-state

  // Replace the below with warning messages
  {
    vector[N_metabolite] conc_change;
    real eps = 1e-6;
    // Should be all 0
    for (cond in 1:N_condition){
      conc_change = S * flux[cond];
      // print(conc_change);
      for (i in 1:N_metabolite){
        if (fabs(conc_change[i]) > eps){
          reject("The steady state assumption must hold. Met ", i,
          "concentration changed ", conc_change[i]);
        }
      }
    }
  }
  // Check that the x values follow steady state
  // {
  //   vector[N_metabolite] conc_change;
  //   matrix[N_metabolite, N_reaction] mod_s;
  //   matrix[N_metabolite, N_metabolite + N_exchange] s_total;
  //   real eps = 1e-6;
  //   // Should be all 0
  //   for (cond in 1:N_condition){
  //     // Determine the final matrix to be solved
  //     mod_s = S;
  //     // This could be done with the left/right multiply function maybe?
  //     mod_s[:, ix_enzyme] = mod_s[:, ix_enzyme] .* rep_matrix((b[cond] .* enzyme[cond])', N_metabolite);
  //     s_total = mod_s * s_gamma;
  //     // Now calculate the conc change
  //     conc_change = s_total * x[cond];
  //     // print(conc_change);
  //     for (i in 1:N_metabolite){
  //       if (fabs(conc_change[i]) > eps){
  //         reject("The steady state assumption must hold. Met ", i,
  //         "concentration changed ", conc_change[i]);
  //       }
  //     }
  //   }
  // }
}
