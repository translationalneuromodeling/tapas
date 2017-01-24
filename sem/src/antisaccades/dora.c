/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "antisaccades.h"

// TODO later, wald and lognorm are place holders!!

double
dora_llh_gamma(double t, int a, DORA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kinv = params.kinv;
    double tinv = params.tinv;
    double kva = params.kva;
    double tva = params.tva;
    double kvp = params.kvp;
    double tvp = params.tvp;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = -INFINITY;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0/t0);
                break;
        }
    }

	switch ( a )
	{
		case PROSACCADE:
			// Account for an early prosaccade
		    fllh = gamma_lpdf(t, kinv, tinv);
    		fllh += log(gsl_sf_gamma_inc_Q(ks, t/ts ));

		    if ( t > da )
			{
        		fllh += log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva ));
        		fllh += log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp ));
			}
			
			// Account for a late prosaccade

			if ( t < da )
			{
				sllh = -INFINITY; //LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kvp, tvp) 
                    + log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					    + gsl_sf_gamma_inc_Q(kinv, t / tinv)) 
                    + 0; //LN_P_LATE_NOT_OUTLIER;
			}
		
			fllh = log(exp(fllh) + exp(sllh)) + log(1 - p0);
			break;
		case ANTISACCADE:
			if ( t < da )
			{
				sllh = -INFINITY;//LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kva, tva) 
                    + log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					+ gsl_sf_gamma_inc_Q(kinv, t / tinv))
                    + 0;//LN_P_LATE_NOT_OUTLIER;
			}
			fllh = sllh + log(1 - p0);
			break;
	}
	
	return fllh;
}

double
dora_llh_mixedgamma(double t, int a, DORA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kinv = params.kinv;
    double tinv = params.tinv;
    double kva = params.kva;
    double tva = params.tva;
    double kvp = params.kvp;
    double tvp = params.tvp;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = -INFINITY;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0 / t0);
                break;
        }
    }

	switch ( a )
	{
		case PROSACCADE:
			// Account for an early prosaccade
		    fllh = invgamma_lpdf(t, kinv, tinv);
    		fllh += log(gsl_sf_gamma_inc_P(ks, ts / t ));

		    if ( t > da )
			{
        		fllh += log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva ));
        		fllh += log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp ));
			}
			
			// Account for a late prosaccade

			if ( t < da )
			{
				sllh = -INFINITY;//LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kvp, tvp) 
                    + log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva)) 
                    + log(ninvgamma_gslint(t, kinv, ks, tinv, ts)
    					+ gsl_sf_gamma_inc_P(kinv, tinv / t)) 
                    + 0;//LN_P_LATE_NOT_OUTLIER;
			}
		
			fllh = log(exp(fllh) + exp(sllh)) + log(1 - p0);
			break;
		case ANTISACCADE:
			if ( t < da )
			{
				sllh = -INFINITY; //LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kva, tva) 
                    + log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp)) 
                    + log(ninvgamma_gslint(t, kinv, ks, tinv, ts)
					    + gsl_sf_gamma_inc_P(kinv, tinv / t))
                    + 0; //LN_P_LATE_NOT_OUTLIER;
			}
			fllh = sllh + log(1 - p0);
			break;
	}
	
	return fllh;

}

double
dora_llh_invgamma(double t, int a, DORA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kinv = params.kinv;
    double tinv = params.tinv;
    double kva = params.kva;
    double tva = params.tva;
    double kvp = params.kvp;
    double tvp = params.tvp;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = -INFINITY;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0/t0);
                break;
        }
    }

	switch ( a )
	{
		case PROSACCADE:
			// Account for an early prosaccade
		    fllh = invgamma_lpdf(t, kinv, tinv);
    		fllh += log(gsl_sf_gamma_inc_P(ks, ts / t ));

		    if ( t > da )
			{
        		fllh += log(gsl_sf_gamma_inc_P(kva, tva / (t - da)));
        		fllh += log(gsl_sf_gamma_inc_P(kvp, tvp / (t - da)));
			}
			
			// Account for a late prosaccade

			if ( t < da )
			{
				sllh = -INFINITY; //LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = invgamma_lpdf(t - da, kvp, tvp) 
                    + log(gsl_sf_gamma_inc_P(kva, tva / (t - da))) 
                    + log(ninvgamma_gslint(t, kinv, ks, tinv, ts)
    					+ gsl_sf_gamma_inc_P(kinv, tinv / t))
                    + 0; // LN_P_LATE_NOT_OUTLIER;
			}
		
			fllh = log(exp(fllh) + exp(sllh)) + log(1 - p0);
			break;
		case ANTISACCADE:
			if ( t < da )
			{
				sllh = -INFINITY; // LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = invgamma_lpdf(t - da, kva, tva) 
                    + log(gsl_sf_gamma_inc_P(kvp, tvp / (t - da))) 
                    + log(ninvgamma_gslint(t, kinv, ks, tinv, ts)
					    + gsl_sf_gamma_inc_P(kinv, tinv / t))
                    + 0; // LN_P_LATE_NOT_OUTLIER;
			}
			fllh = sllh + log(1 - p0);
			break;
	}
	
	return fllh;

}


double
dora_llh_lognorm(double t, int a, DORA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kinv = params.kinv;
    double tinv = params.tinv;
    double kva = params.kva;
    double tva = params.tva;
    double kvp = params.kvp;
    double tvp = params.tvp;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = -INFINITY;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0/t0);
                break;
        }
    }

	switch ( a )
	{
		case PROSACCADE:
			// Account for an early prosaccade
		    fllh = gamma_lpdf(t, kinv, tinv);
    		fllh += log(gsl_sf_gamma_inc_Q(ks, t/ts ));

		    if ( t > da )
			{
        		fllh += log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva ));
        		fllh += log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp ));
			}
			
			// Account for a late prosaccade

			if ( t < da )
			{
				sllh = -INFINITY; //LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kvp, tvp) 
                    + log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					    + gsl_sf_gamma_inc_Q(kinv, t / tinv)) 
                    + 0; //LN_P_LATE_NOT_OUTLIER;
			}
		
			fllh = log(exp(fllh) + exp(sllh)) + log(1 - p0);
			break;
		case ANTISACCADE:
			if ( t < da )
			{
				sllh = -INFINITY;//LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kva, tva) 
                    + log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					+ gsl_sf_gamma_inc_Q(kinv, t / tinv))
                    + 0;//LN_P_LATE_NOT_OUTLIER;
			}
			fllh = sllh + log(1 - p0);
			break;
	}
	
	return fllh;
}
double
dora_llh_wald(double t, int a, DORA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kinv = params.kinv;
    double tinv = params.tinv;
    double kva = params.kva;
    double tva = params.tva;
    double kvp = params.kvp;
    double tvp = params.tvp;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = -INFINITY;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0/t0);
                break;
        }
    }

	switch ( a )
	{
		case PROSACCADE:
			// Account for an early prosaccade
		    fllh = gamma_lpdf(t, kinv, tinv);
    		fllh += log(gsl_sf_gamma_inc_Q(ks, t/ts ));

		    if ( t > da )
			{
        		fllh += log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva ));
        		fllh += log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp ));
			}
			
			// Account for a late prosaccade

			if ( t < da )
			{
				sllh = -INFINITY; //LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kvp, tvp) 
                    + log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					    + gsl_sf_gamma_inc_Q(kinv, t / tinv)) 
                    + 0; //LN_P_LATE_NOT_OUTLIER;
			}
		
			fllh = log(exp(fllh) + exp(sllh)) + log(1 - p0);
			break;
		case ANTISACCADE:
			if ( t < da )
			{
				sllh = -INFINITY;//LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kva, tva) 
                    + log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					+ gsl_sf_gamma_inc_Q(kinv, t / tinv))
                    + 0;//LN_P_LATE_NOT_OUTLIER;
			}
			fllh = sllh + log(1 - p0);
			break;
	}
	
	return fllh;
}

double
dora_llh_later(double t, int a, DORA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kinv = params.kinv;
    double tinv = params.tinv;
    double kva = params.kva;
    double tva = params.tva;
    double kvp = params.kvp;
    double tvp = params.tvp;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = -INFINITY;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0/t0);
                break;
        }
    }

	switch ( a )
	{
		case PROSACCADE:
			// Account for an early prosaccade
		    fllh = gamma_lpdf(t, kinv, tinv);
    		fllh += log(gsl_sf_gamma_inc_Q(ks, t/ts ));

		    if ( t > da )
			{
        		fllh += log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva ));
        		fllh += log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp ));
			}
			
			// Account for a late prosaccade

			if ( t < da )
			{
				sllh = -INFINITY; //LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kvp, tvp) 
                    + log(gsl_sf_gamma_inc_Q(kva, (t - da)/tva)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					    + gsl_sf_gamma_inc_Q(kinv, t / tinv)) 
                    + 0; //LN_P_LATE_NOT_OUTLIER;
			}
		
			fllh = log(exp(fllh) + exp(sllh)) + log(1 - p0);
			break;
		case ANTISACCADE:
			if ( t < da )
			{
				sllh = -INFINITY;//LN_P_LATE_OUTLIER;
			} else 
			{
				sllh = gamma_lpdf(t - da, kva, tva) 
                    + log(gsl_sf_gamma_inc_Q(kvp, (t - da)/tvp)) 
                    + log(ngamma_gslint(t, kinv, ks, tinv, ts)
					+ gsl_sf_gamma_inc_Q(kinv, t / tinv))
                    + 0;//LN_P_LATE_NOT_OUTLIER;
			}
			fllh = sllh + log(1 - p0);
			break;
	}
	
	return fllh;
}
