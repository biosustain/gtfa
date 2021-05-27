/* Custom functions */

// packing things into algebra solver variables...
int[] get_x_i(int N_metabolite,
              int N_reaction,
              int[] enzyme_to_reaction,
              int[] ix_b_free,
              int[] ix_b_bound){
  int N_enzyme = size(enzyme_to_reaction);
  int N_b_free = size(ix_b_free);
  return append_array({N_metabolite, N_enzyme, N_reaction, N_b_free},
                      append_array(append_array(enzyme_to_reaction, ix_b_free), ix_b_bound));
}

real[] get_x_r(matrix S){return to_array_1d(S);}

vector get_theta(vector dgf, vector b_free, vector enzyme, vector metabolite){
  return append_row(dgf, append_row(b_free, append_row(enzyme, metabolite)));
}

int get_N_x_i(int N_enzyme){return 4 + N_enzyme *2;}

int get_N_x_r(real[] x_r){return size(x_r);}

int get_N_theta(int N_metabolite, int N_b_free, int N_enzyme){
  return N_metabolite * 2 + N_b_free + N_enzyme;
}

// unpacking algebra solver variables...
int extract_N_metabolite(int[] x_i){return x_i[1];}
int extract_N_enzyme(int[] x_i){return x_i[2];}
int extract_N_reaction(int[] x_i){return x_i[3];}
int extract_N_b_free(int[] x_i){return x_i[4];}
int[] extract_enzyme_to_reaction(int[] x_i){return x_i[5:5+extract_N_enzyme(x_i)];}
int[] extract_ix_b_free(int[] x_i){
  int start = 5 + extract_N_enzyme(x_i);
  int end = start + extract_N_b_free(x_i);
  return x_i[start:end];
}
int[] extract_ix_b_bound(int[] x_i){
  int N_b_bound = extract_N_enzyme(x_i) - extract_N_b_free(x_i);
  int start = 5 + extract_N_enzyme(x_i) + extract_N_b_free(x_i);
  int end = start + N_b_bound;
  return x_i[start:end];
}
matrix extract_S(real[] x_r, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  int N_enzyme = extract_N_enzyme(x_i);
  return to_matrix(x_r, N_metabolite, N_enzyme);
}
vector extract_dgf(vector theta, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  return theta[1:N_metabolite];
}
vector extract_b_free(vector theta, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  int N_enzyme = extract_N_enzyme(x_i);
  return theta[N_metabolite + 1: N_metabolite + N_enzyme];
}
vector extract_enzyme(vector theta, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  int N_enzyme = extract_N_enzyme(x_i);
  return theta[N_metabolite + N_enzyme : N_metabolite + N_enzyme * 2];
}
vector extract_metabolite(vector theta, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  int N_enzyme = extract_N_enzyme(x_i);
  return theta[N_metabolite + N_enzyme * 2 :];
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

  vector[cols(S)] dgr = get_dgr(S,
                                extract_dgf(theta, x_i),
                                extract_metabolite(theta, x_i));
  vector[cols(S)] b = get_b(extract_b_free(theta, x_i),
                            b_bound,
                            extract_ix_b_free(x_i),
                            extract_ix_b_bound(x_i));
  return dgr .* extract_enzyme(theta, x_i) .* b;
}

vector steady_state(vector b_bound, vector theta, real[] x_r, int[] x_i){
  /* Main function - returns the steady state value of the bound b
     parameters. */

  matrix[extract_N_metabolite(x_i), extract_N_enzyme(x_i)] S = extract_S(x_r, x_i);
  array[rows(b_bound)] int ix_b_bound = extract_ix_b_bound(x_i);
  return (S * get_flux(S, b_bound, theta, x_i))[ix_b_bound];
}
