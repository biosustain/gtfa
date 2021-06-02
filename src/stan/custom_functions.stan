/* Custom functions */

// packing things into algebra solver variables...
vector get_theta(vector dgf, vector b_free, vector drain, vector enzyme, vector metabolite){
  return append_row(dgf, append_row(b_free, append_row(drain, append_row(enzyme, metabolite))));
}

// unpacking algebra solver variables...
int extract_N_metabolite(int[] x_i){return x_i[1];}
int extract_N_enzyme(int[] x_i){return x_i[2];}
int extract_N_reaction(int[] x_i){return x_i[3];}
int extract_N_drain(int[] x_i){return extract_N_reaction(x_i) - extract_N_enzyme(x_i);}
int extract_N_b_free(int[] x_i){return x_i[4];}
int extract_N_b_bound(int[] x_i){return x_i[5];}
int[] extract_reaction_to_enzyme(int[] x_i){
  int start = 6;
  int N = extract_N_reaction(x_i);
  return x_i[start: start + N - 1];
}
int[] extract_reaction_to_drain(int[] x_i){
  int start = 6 + extract_N_reaction(x_i);
  int N = extract_N_reaction(x_i);
  return x_i[start: start + N - 1];
}
int[] extract_ix_b_free(int[] x_i){
  int start = 6 + extract_N_reaction(x_i) + extract_N_reaction(x_i);
  int N = extract_N_b_free(x_i);
  return x_i[start: start + N - 1];
}
int[] extract_ix_b_bound(int[] x_i){
  int start = 6 + extract_N_reaction(x_i) + extract_N_reaction(x_i) + extract_N_b_free(x_i);
  int N = extract_N_b_bound(x_i);
  return x_i[start: start + N - 1];
}
matrix extract_S(real[] x_r, int[] x_i){
  return to_matrix(x_r, extract_N_metabolite(x_i), extract_N_reaction(x_i));
}
vector extract_dgf(vector theta, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  return theta[1:N_metabolite];
}
vector extract_b_free(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i);
  int N = extract_N_b_free(x_i);
  return theta[start: start + N - 1];
}
vector extract_drain(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i) + extract_N_b_free(x_i);
  int N = extract_N_drain(x_i);
  return theta[start: start + N - 1];
}
vector extract_enzyme(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i) + extract_N_b_free(x_i) + extract_N_drain(x_i);
  int N = extract_N_enzyme(x_i);
  return theta[start: start + N - 1];
}
vector extract_metabolite(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i) + extract_N_b_free(x_i) + extract_N_drain(x_i) + extract_N_enzyme(x_i);
  int N = extract_N_metabolite(x_i);
  return theta[start: start + N - 1];
}

vector get_b(vector b_free, vector b_bound, int[] ix_b_free, int[] ix_b_bound){
  /* Get a vector of b parameters. */

  vector[rows(b_free) + rows(b_bound)] out;
  out[ix_b_bound] = b_bound;
  out[ix_b_free] = b_free;
  return out;
}

vector get_dgr(matrix S, vector dgf, vector metabolite){
  /* Get the delta G of reaction from delta G of formation and metabolite
     concentrations. */

  real RT = 0.008314 * 298.15;
  return S' * (dgf + RT * log(metabolite));
}

vector get_flux(matrix S, vector b_bound, vector theta, int[] x_i){
  /* Get the flux from the stoichiometric matrix, concentration of bound bs and
     theta and x_i parameters. */

  int N_reaction = cols(S);
  int N_enzyme = extract_N_enzyme(x_i);
  int N_drain = extract_N_drain(x_i);
  vector[N_reaction] out;
  array[N_reaction] int reaction_to_enzyme = extract_reaction_to_enzyme(x_i);
  array[N_reaction] int reaction_to_drain = extract_reaction_to_drain(x_i);
  vector[N_reaction] dgr = get_dgr(S,
                                   extract_dgf(theta, x_i),
                                   extract_metabolite(theta, x_i));
  vector[N_enzyme] b = get_b(extract_b_free(theta, x_i),
                             b_bound,
                             extract_ix_b_free(x_i),
                             extract_ix_b_bound(x_i));
  vector[N_drain] drain = extract_drain(theta, x_i);
  vector[N_enzyme] enzyme = extract_enzyme(theta, x_i);
  for (r in 1:cols(S)){
    int d = reaction_to_drain[r];
    int e = reaction_to_enzyme[r];
    if ((d != 0) && (e != 0)) {
      reject("Both drain and enzyme index are non-zero for reaction ", r);
    }
    else if ((d == 0) && (e == 0)) {
      reject("Both drain nor reaction index are zero for reaction ", r);
    }
    else {
      out[r] = d != 0 ? drain[d] : -dgr[r] * enzyme[e] * b[e];
    }
  }
  return out;
}

vector steady_state(vector b_bound, vector theta, real[] x_r, int[] x_i){
  /* Main function - returns the steady state value of the bound b
     parameters. */

  matrix[extract_N_metabolite(x_i), extract_N_enzyme(x_i)] S = extract_S(x_r, x_i);
  array[rows(b_bound)] int ix_b_bound = extract_ix_b_bound(x_i);
  return S * get_flux(S, exp(b_bound), theta, x_i);
}
