use crate::data::{Data, Row};

struct Parameters {
    s: f64,
    tau: f64,
    mu1: f64,
    mu2: f64,
    gamma1: f64,
    gamma2: f64,
}

struct Chain {
    data: Data,
    parameters: Parameters,
}

impl Chain {
    fn new(data: Data) -> Self {
        let parameters = Parameters {
            s: 1.0,
            tau: 0.5,
            mu1: 0.0,
            mu2: 0.0,
            gamma1: 0.0,
            gamma2: 0.0,
        };
        Self { data, parameters }
    }

    fn l(&self) -> f64 {
        self.data.iter().map(|row| row_likelihood(&row, &self.parameters)).product()
    }
}

// essentially the numerator of the Bayes' formula: normal likelihood * prior on sigma^2 
// other priors are uniform, so they don't play a role here
fn row_likelihood(r: &Row, p: &Parameters) -> f64 {
    match r.group {
        1 => { pnorm(r.x1, p.mu1, p.s) * pnorm(r.x2, p.mu2, p.s)/p.s },
        2 => { pnorm(r.x1, p.gamma1, p.s) * pnorm(r.x2, p.gamma2, p.s)/p.s   }
        3 => { pnorm(r.x1, 0.5*p.mu1 + 0.5*p.gamma1, p.s) * pnorm(r.x2, 0.5*p.mu2 + 0.5*p.gamma2, p.s)/p.s  }
        4 => { pnorm(r.x1, p.tau*p.mu1 + (1. - p.tau)*p.gamma1, p.s) * pnorm(r.x2, p.tau*p.mu2 + (1. - p.tau)*p.gamma2, p.s)/p.s  }
        _ => unreachable!()
    }
}

// normal likelihood  
fn pnorm(x:f64, mu: f64, s: f64) -> f64 {
    (-(x - mu).powi(2)/2.0/s).exp()/s.sqrt()
}