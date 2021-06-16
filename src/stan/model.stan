functions {
#include custom_functions.stan
  vector get_dgr(matrix S, vector dgf, vector log_metabolite){
    /* Get the delta G of reaction from delta G of formation and metabolite
       concentrations. */
    real RT = 0.008314 * 298.15;
    return S' * (dgf + RT * log_metabolite);
  }
  vector v(vector log_metabolite, vector theta, real[] x_r, int[] x_i){
    /* Get the flux for given parameters set and log metabolite concentrations. */
    int N_reaction = extract_N_reaction(x_i);
    int N_enzyme = extract_N_enzyme(x_i);
    int N_tranport = extract_N_transport(x_i);
    vector[N_reaction] out;
    array[N_reaction] int reaction_to_enzyme = extract_reaction_to_enzyme(x_i);
    array[N_reaction] int reaction_to_transport = extract_reaction_to_transport(x_i);
    vector[N_reaction] dgr = get_dgr(extract_S(x_r, x_i), extract_dgf(theta, x_i), log_metabolite);
    vector[N_enzyme] b = extract_b(theta, x_i);
    vector[N_tranport] transport = extract_transport(theta, x_i);
    vector[N_enzyme] enzyme = extract_enzyme(theta, x_i);
    for (r in 1:N_reaction){
      int t = reaction_to_transport[r];
      int e = reaction_to_enzyme[r];
      if ((t != 0) && (e != 0)) {
        reject("Both transport and enzyme index are non-zero for reaction ", r);
      }
      else if ((t == 0) && (e == 0)) {
        reject("Neither transport nor reaction index are zero for reaction ", r);
      }
      else {
        out[r] = t != 0 ? transport[t] : -dgr[r] * enzyme[e] * b[e];
      }
    }
    return out;
  }
  vector Sv(vector log_metabolite, vector theta, real[] x_r, int[] x_i){
    /* Multiply the stoichiometric matrix by the flux to get the rate of change
       of compound concentrations. */ 
    matrix[extract_N_metabolite(x_i), extract_N_enzyme(x_i)] S = extract_S(x_r, x_i);
    return S * v(log_metabolite, theta, x_r, x_i);
  }
}
data {
  // network properties
  int<lower=1> N_metabolite;
  int<lower=1> N_transport;
  int<lower=1> N_enzyme;
  int<lower=1> N_reaction;
  matrix[N_metabolite, N_reaction] S;
  array[N_reaction] int<lower=0,upper=N_enzyme> reaction_to_enzyme;  // zero if no enzyme
  array[N_reaction] int<lower=0,upper=N_transport> reaction_to_transport;    // zero if no drain
  array[N_enzyme] int<lower=1,upper=N_reaction> enzyme_to_reaction;
  array[N_transport] int<lower=1,upper=N_reaction> transport_to_reaction;
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
  array[2] vector[N_metabolite] prior_dgf;
  array[2, N_condition] vector[N_transport] prior_transport;
  array[2, N_condition] vector[N_enzyme] prior_enzyme;
  array[2, N_condition] vector[N_enzyme] prior_b;
  // config
  array[N_condition] vector[N_metabolite] log_metabolite_guess;
  int<lower=0,upper=1> likelihood;
  real rel_tol;
  real function_tol;
  int max_num_steps;
}
transformed data {
  array[rows(S)*cols(S)] real x_r = to_array_1d(S);
  array[3 + 3 * N_reaction] int x_i = get_x_i(N_metabolite, N_enzyme, N_reaction,
                                              reaction_to_enzyme, reaction_to_transport,
                                              enzyme_to_reaction, transport_to_reaction);
                                     
}
parameters {
  vector[N_metabolite] dgf_z;
  array[N_condition] vector[N_transport] transport_z;
  array[N_condition] vector<lower=0>[N_enzyme] enzyme;
  array[N_condition] vector<lower=0>[N_enzyme] b;
}
transformed parameters {
  vector[N_metabolite] dgf = prior_dgf[1] + dgf_z .* prior_dgf[2];
  array[N_condition] vector[N_transport] transport;
  array[N_condition] vector[N_reaction] flux;
  array[N_condition] vector[N_metabolite] log_metabolite;
  array[N_condition] vector[N_reaction] dgr;
  for (c in 1:N_condition){
    transport[c] = prior_transport[1, c] + transport_z[c] .* prior_transport[2, c];
    int N_theta = N_metabolite + N_enzyme + N_transport + N_enzyme;
    vector[N_theta] theta = get_theta(dgf, b[c], transport[c], enzyme[c]);
    log_metabolite[c] = algebra_solver_newton(Sv,
                                              log_metabolite_guess[c],
                                              theta,
                                              x_r,
                                              x_i,
                                              rel_tol,
                                              function_tol,
                                              max_num_steps);
    flux[c] = v(log_metabolite[c], theta, x_r, x_i);
    dgr[c] = get_dgr(S, dgf, log_metabolite[c]);
  }
}
model {
  dgf_z ~ std_normal();
  for (c in 1:N_condition){
    transport_z[c] ~ std_normal();
    transport[c] ~ normal(prior_transport[1, c], prior_transport[2, c]);
    enzyme[c] ~ lognormal(prior_enzyme[1, c], prior_enzyme[2, c]);
    b[c] ~ lognormal(prior_b[1, c], prior_b[2, c]);
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
      y_metabolite[n] ~ lognormal(log_metabolite[c, m], sigma_metabolite[n]);
    }
    for (n in 1:N_y_flux){
      int c = condition_y_flux[n];
      int r = reaction_y_flux[n];
      y_flux[n] ~ lognormal(flux[c, r], sigma_flux[n]);
    }
  }
}

