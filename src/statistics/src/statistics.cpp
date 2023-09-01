#include "../include/statistics.hpp"

Statistics::Statistics(){
    jl_init();
    jl_eval_string("import Pkg; Pkg.add(\"Distributions\")");
    jl_eval_string("using Distributions");
}
Statistics::~Statistics(){
    jl_exit(0);
}
double GetChi2(double sigma, int degree){
    jl_function_t *pcquantile = jl_get_function(jl_base_module, "cquantile");
    jl_function_t *pchisq = jl_get_function(jl_base_module, "Chisq");
    jl_value_t *psigma = jl_box_float64(sigma);
    jl_value_t *pdegree = jl_box_int32(degree);
    jl_function_t *pdist = jl_call1(pchisq,pdegree);
    jl_value_t *chi2 = jl_call2(pcquantile,pdist,psigma);
    return jl_unbox_float64(chi2);
}