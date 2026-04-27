import torch
from torch import sqrt
from torch.distributions import Normal

def analyze_ab_test(control_outcomes: list, treatment_outcomes: list,
confidence_level: float = 0.95, min_detectable_effect: float = 0.02) ->dict:
    """
    Analyze A/B test results for model comparison with statistical rigor.
    
    Args:
        control_outcomes: List of binary outcomes (0 or 1) for control group
        treatment_outcomes: List of binary outcomes (0 or 1) for treatment group
        confidence_level: Confidence level for statistical tests (default 0.95)
        min_detectable_effect: Minimum absolute effect size considered practically significant
    
    Returns:
        dict with statistical analysis results and recommendation
    """
    control_len,treatment_len=len(control_outcomes),len(treatment_outcomes)
    if control_len == 0 or treatment_len == 0:
        return {}
    
    control_rate,treatment_rate=sum(control_outcomes)/control_len,sum(treatment_outcomes)/treatment_len
    absolute_lift=treatment_rate-control_rate
    absolute_lift=round(absolute_lift, 4)
    relative_lift=(treatment_rate-control_rate)/control_rate
    

    p_pool=(control_rate*control_len+treatment_rate*treatment_len)/(control_len+treatment_len)
    z = (treatment_rate-control_rate)/sqrt(torch.tensor(p_pool*(1-p_pool)*(control_len+treatment_len)/(control_len*treatment_len))).item()
    z = round(z, 4)
    se = sqrt(torch.tensor(control_rate*(1-control_rate)/control_len+treatment_rate*(1-treatment_rate)/treatment_len)).item()
    alpha = 1-confidence_level
    dist = Normal(loc=0.0, scale=1.0)
    z_crit = dist.icdf(torch.tensor(1-alpha/2)).item()
    z_low, z_high = (treatment_rate-control_rate)-z_crit*se,(treatment_rate-control_rate)+z_crit*se
    p_value = 2 * (1 - dist.cdf(torch.tensor(abs(z))).item()) 
    stat_sig= p_value<alpha
    pract_sig = absolute_lift > min_detectable_effect
    beta=0.84
    required_sample_size=2*((alpha+beta)/min_detectable_effect)**2\
                        *p_pool*(1-p_pool)

    recommendation=""

    if stat_sig:
        if pract_sig:
            if absolute_lift > 0:
                recommendation = "launch_treatment"
        else:
            recommendation = "keep_control"
    else:
        recommendation = "continue_testing"
    
    return {"control_rate":control_rate,
            "treatment_rate":treatment_rate,
            "absolute_lift":absolute_lift,
            "relative_lift":relative_lift,
            "z_statistic":z,
            "p_value":p_value,
            "confidence_interval":(z_low, z_high),
            "statistically_significant":stat_sig,
            "practically_significant":pract_sig,
            "required_sample_size":required_sample_size,
            "recommendation":recommendation
            }