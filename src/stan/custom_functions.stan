/* Custom functions */

vector get_theta(vector dgf, vector b, vector drain, vector enzyme){
  return append_row(dgf, append_row(b, append_row(drain, enzyme)));
}
int[] get_x_i(int N_metabolite,
                  int N_enzyme,
                  int N_reaction,
                  int[] reaction_to_enzyme,
                  int[] reaction_to_transport,
                  int[] enzyme_to_reaction,
                  int[] transport_to_reaction){
  return append_array({N_metabolite, N_enzyme, N_reaction},
                      append_array(reaction_to_enzyme,
                                   append_array(reaction_to_transport,
                                                append_array(enzyme_to_reaction,
                                                             transport_to_reaction))));
}
int extract_N_metabolite(int[] x_i){return x_i[1];}
int extract_N_enzyme(int[] x_i){return x_i[2];}
int extract_N_reaction(int[] x_i){return x_i[3];}
int extract_N_transport(int[] x_i){return extract_N_reaction(x_i) - extract_N_enzyme(x_i);}
int[] extract_reaction_to_enzyme(int[] x_i){
  int start = 4;
  int N = extract_N_reaction(x_i);
  return x_i[start: start + N - 1];
}
int[] extract_reaction_to_transport(int[] x_i){
  int start = 4 + extract_N_reaction(x_i);
  int N = extract_N_reaction(x_i);
  return x_i[start: start + N - 1];
}
int[] extract_enzyme_to_reaction(int[] x_i){
  int start = 4 + 2 * extract_N_reaction(x_i);
  int N = extract_N_enzyme(x_i);
  return x_i[start: start + N - 1];
}
int[] extract_transport_to_reaction(int[] x_i){
  int start = 4 + 2 * extract_N_reaction(x_i) + extract_N_enzyme(x_i);
  int N = extract_N_transport(x_i);
  return x_i[start: start + N - 1];
}
matrix extract_S(real[] x_r, int[] x_i){
  return to_matrix(x_r, extract_N_metabolite(x_i), extract_N_reaction(x_i));
}
vector extract_dgf(vector theta, int[] x_i){
  int N_metabolite = extract_N_metabolite(x_i);
  return theta[1:N_metabolite];
}
vector extract_b(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i);
  int N = extract_N_enzyme(x_i);
  return theta[start: start + N - 1];
}
vector extract_transport(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i) + extract_N_enzyme(x_i);
  int N = extract_N_transport(x_i);
  return theta[start: start + N - 1];
}
vector extract_enzyme(vector theta, int[] x_i){
  int start = 1 + extract_N_metabolite(x_i) + extract_N_enzyme(x_i) + extract_N_transport(x_i);
  int N = extract_N_enzyme(x_i);
  return theta[start: start + N - 1];
}
