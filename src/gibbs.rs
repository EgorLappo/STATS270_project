use crate::data::{Data, Row};
use rand::prelude::*;
use rand::distributions::Distribution;
use statrs::distribution::{Uniform, Normal};

static SPROPSD: f64 = 0.1;
static MEANPROPSD: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct Parameters {
    s: f64,
    tau: f64,
    mu1: f64,
    mu2: f64,
    gamma1: f64,
    gamma2: f64,
}

pub struct Chain {
    data: Data,
    parameters: Parameters,
}

impl Chain {
    pub fn new(data: Data) -> Self {
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

    pub fn run(&mut self, n_burnin: usize, n_samples: usize, seed: u64) -> Vec<Parameters> {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..n_burnin {
            self.step(&mut rng);
        }

        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            self.step(&mut rng);
            samples.push(self.parameters.clone());
        }

        samples
    }

    fn step(&mut self, rng: &mut StdRng) {
        self.update_s(rng);
        self.update_tau(rng);
        self.update_mu(rng);
        self.update_gamma(rng);
    }


    fn update_s(&mut self, rng: &mut StdRng) {
        unimplemented!()
    }

    fn update_tau(&mut self, rng: &mut StdRng) {
        unimplemented!()
    }

    fn update_mu(&mut self, rng: &mut StdRng) {
        unimplemented!()
    }

    fn update_gamma(&mut self, rng: &mut StdRng) { 
        unimplemented!()
    }
}

// normal likelihood  
fn pnorm(x:f64, mu: f64, s: f64) -> f64 {
    (-(x - mu).powi(2)/2.0/s).exp()/s.sqrt()
}
    
impl Parameters {
    pub fn summary(ps: Vec<Parameters>) -> String {
        // compute mean of each parameter
        let n = ps.len() as f64;

        let mut s = 0.0;
        let mut tau = 0.0;
        let mut mu1 = 0.0;
        let mut mu2 = 0.0;
        let mut gamma1 = 0.0;
        let mut gamma2 = 0.0;

        let mut s2 = 0.0;
        let mut tau2 = 0.0;
        let mut mu12 = 0.0;
        let mut mu22 = 0.0;
        let mut gamma12 = 0.0;
        let mut gamma22 = 0.0;

        for p in &ps {
            s += p.s;
            tau += p.tau;
            mu1 += p.mu1;
            mu2 += p.mu2;
            gamma1 += p.gamma1;
            gamma2 += p.gamma2;
        }
        
        s /= n;
        tau /= n;
        mu1 /= n;
        mu2 /= n;
        gamma1 /= n;
        gamma2 /= n;

        // compute 5th and 95th quantiles of each parameter
        let mut s_vec = Vec::new();
        let mut tau_vec = Vec::new();
        let mut mu1_vec = Vec::new();
        let mut mu2_vec = Vec::new();
        let mut gamma1_vec = Vec::new();
        let mut gamma2_vec = Vec::new();

        for p in &ps {
            s_vec.push(p.s);
            tau_vec.push(p.tau);
            mu1_vec.push(p.mu1);
            mu2_vec.push(p.mu2);
            gamma1_vec.push(p.gamma1);
            gamma2_vec.push(p.gamma2);
        }

        s_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tau_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mu1_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mu2_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gamma1_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gamma2_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut s_5 = s_vec[(n*0.05) as usize];
        let mut tau_5 = tau_vec[(n*0.05) as usize];
        let mut mu1_5 = mu1_vec[(n*0.05) as usize];
        let mut mu2_5 = mu2_vec[(n*0.05) as usize];
        let mut gamma1_5 = gamma1_vec[(n*0.05) as usize];
        let mut gamma2_5 = gamma2_vec[(n*0.05) as usize];

        let mut s_95 = s_vec[(n*0.95) as usize];
        let mut tau_95 = tau_vec[(n*0.95) as usize];
        let mut mu1_95 = mu1_vec[(n*0.95) as usize];
        let mut mu2_95 = mu2_vec[(n*0.95) as usize];
        let mut gamma1_95 = gamma1_vec[(n*0.95) as usize];
        let mut gamma2_95 = gamma2_vec[(n*0.95) as usize];

        // format mean and quantiles into a summary to print out

        format!("s: {:.3} [{:.3}, {:.3}]
                 tau: {:.3} [{:.3}, {:.3}]
                 mu1: {:.3} [{:.3}, {:.3}]
                 mu2: {:.3} [{:.3}, {:.3}]
                 gamma1: {:.3} [{:.3}, {:.3}]
                 gamma2: {:.3} [{:.3}, {:.3}]", s, s_5, s_95, tau, tau_5, tau_95, mu1, mu1_5, mu1_95, mu2, mu2_5, mu2_95, gamma1, gamma1_5, gamma1_95, gamma2, gamma2_5, gamma2_95)

    }
}
