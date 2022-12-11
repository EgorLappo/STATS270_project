use crate::data::{Data};
use rand::prelude::*;
use rand::distributions::Distribution;
use serde::{Serialize, Deserialize};
use statrs::distribution::{Normal, ChiSquared};

static SPROPSD: f64 = 0.1;
static MEANPROPSD: f64 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
            s: 0.06,
            tau: 0.8,
            mu1: -1.4,
            mu2: -0.7,
            gamma1: -0.2,
            gamma2: 0.3,
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
        let N = self.data.len() as f64;
        let chisq = ChiSquared::new(2.*N).unwrap();

        self.parameters.s = 1./chisq.sample(rng);
    }

    fn update_tau(&mut self, rng: &mut StdRng) {
        let n4 = self.data.iter().filter(|r| r.group == 4).count() as f64;

        let x41mean = self.data.iter().filter(|r| r.group == 4).map(|r| r.x1).sum::<f64>()/n4;
        let x42mean = self.data.iter().filter(|r| r.group == 4).map(|r| r.x2).sum::<f64>()/n4;

        let denom = n4 * (self.parameters.mu1 - self.parameters.gamma1).powi(2) 
                  + n4 * (self.parameters.mu2 - self.parameters.gamma2).powi(2);
        let numer = n4 * (self.parameters.mu1 - self.parameters.gamma1)*(x41mean - self.parameters.gamma1) 
        + n4 * (self.parameters.mu2 - self.parameters.gamma2)*(x42mean - self.parameters.gamma2);

        let n = Normal::new(numer/denom, (self.parameters.s/denom).sqrt()).unwrap();

        let mut new_tau: f64 = n.sample(rng);

        while new_tau <= 0.0 || new_tau >= 1.0 {
            new_tau = n.sample(rng);
        }

        self.parameters.tau = new_tau;
    }

    fn update_mu(&mut self, rng: &mut StdRng) {
        let n1 = self.data.iter().filter(|r| r.group == 1).count() as f64;
        let n3 = self.data.iter().filter(|r| r.group == 3).count() as f64;
        let n4 = self.data.iter().filter(|r| r.group == 4).count() as f64;

        let x11mean = self.data.iter().filter(|r| r.group == 1).map(|r| r.x1).sum::<f64>()/n1;
        let x12mean = self.data.iter().filter(|r| r.group == 1).map(|r| r.x2).sum::<f64>()/n1;

        let x31mean = self.data.iter().filter(|r| r.group == 3).map(|r| r.x1).sum::<f64>()/n3;
        let x32mean = self.data.iter().filter(|r| r.group == 3).map(|r| r.x2).sum::<f64>()/n3;

        let x41mean = self.data.iter().filter(|r| r.group == 4).map(|r| r.x1).sum::<f64>()/n4;
        let x42mean = self.data.iter().filter(|r| r.group == 4).map(|r| r.x2).sum::<f64>()/n4;

        let denom = n1 + n3*0.25 + n4*self.parameters.tau.powi(2);
        let numer1 = n1*x11mean 
                   + n3*0.25*(2.*x31mean-self.parameters.gamma1) 
                   + n4*self.parameters.tau*(x41mean-(1.-self.parameters.tau)*self.parameters.gamma1);

        let numer2 = n1*x12mean 
                   + n3*0.25*(2.*x32mean-self.parameters.gamma2) 
                   + n4*self.parameters.tau*(x42mean-(1.-self.parameters.tau)*self.parameters.gamma2);

        let n1 = Normal::new(numer1/denom, (self.parameters.s/denom).sqrt()).unwrap();
        let n2 = Normal::new(numer2/denom, (self.parameters.s/denom).sqrt()).unwrap();

        self.parameters.mu1 = n1.sample(rng);
        self.parameters.mu2 = n2.sample(rng);
    }

    fn update_gamma(&mut self, rng: &mut StdRng) { 
        let n2 = self.data.iter().filter(|r| r.group == 2).count() as f64;
        let n3 = self.data.iter().filter(|r| r.group == 3).count() as f64;
        let n4 = self.data.iter().filter(|r| r.group == 4).count() as f64;

        let x21mean = self.data.iter().filter(|r| r.group == 2).map(|r| r.x1).sum::<f64>()/n2;
        let x22mean = self.data.iter().filter(|r| r.group == 2).map(|r| r.x2).sum::<f64>()/n2;

        let x31mean = self.data.iter().filter(|r| r.group == 3).map(|r| r.x1).sum::<f64>()/n3;
        let x32mean = self.data.iter().filter(|r| r.group == 3).map(|r| r.x2).sum::<f64>()/n3;

        let x41mean = self.data.iter().filter(|r| r.group == 4).map(|r| r.x1).sum::<f64>()/n4;
        let x42mean = self.data.iter().filter(|r| r.group == 4).map(|r| r.x2).sum::<f64>()/n4;

        let denom = n2 + n3*0.25 + n4*(1.-self.parameters.tau).powi(2);
        let numer1 = n2*x21mean 
                   + n3*0.25*(2.*x31mean-self.parameters.mu1) 
                   + n4*(1.-self.parameters.tau)*(x41mean-self.parameters.tau*self.parameters.mu1);

        let numer2 = n2*x22mean 
                   + n3*0.25*(2.*x32mean-self.parameters.mu2) 
                   + n4*(1.-self.parameters.tau)*(x42mean-self.parameters.tau*self.parameters.mu2);

        let n1 = Normal::new(numer1/denom, (self.parameters.s/denom).sqrt()).unwrap();
        let n2 = Normal::new(numer2/denom, (self.parameters.s/denom).sqrt()).unwrap();

        self.parameters.gamma1 = n1.sample(rng);
        self.parameters.gamma2 = n2.sample(rng);
    }
}
    
impl Parameters {
    pub fn save_to_csv(ps: &Vec<Parameters>, filename: &str) {
        let mut wtr = csv::Writer::from_path(filename).unwrap();

        for p in ps {
            wtr.serialize(p).unwrap();
        }
    }

    pub fn summary(ps: &Vec<Parameters>) -> String {
        // compute mean of each parameter
        let n = ps.len() as f64;

        let mut s = 0.0;
        let mut tau = 0.0;
        let mut mu1 = 0.0;
        let mut mu2 = 0.0;
        let mut gamma1 = 0.0;
        let mut gamma2 = 0.0;

        for p in ps {
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

        for p in ps {
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

        let s_5 = s_vec[(n*0.05) as usize];
        let tau_5 = tau_vec[(n*0.05) as usize];
        let mu1_5 = mu1_vec[(n*0.05) as usize];
        let mu2_5 = mu2_vec[(n*0.05) as usize];
        let gamma1_5 = gamma1_vec[(n*0.05) as usize];
        let gamma2_5 = gamma2_vec[(n*0.05) as usize];

        let s_95 = s_vec[(n*0.95) as usize];
        let tau_95 = tau_vec[(n*0.95) as usize];
        let mu1_95 = mu1_vec[(n*0.95) as usize];
        let mu2_95 = mu2_vec[(n*0.95) as usize];
        let gamma1_95 = gamma1_vec[(n*0.95) as usize];
        let gamma2_95 = gamma2_vec[(n*0.95) as usize];

        // format mean and quantiles into a summary to print out

        format!("s: {:.3} [{:.3}, {:.3}]\ntau: {:.3} [{:.3}, {:.3}]\nmu1: {:.3} [{:.3}, {:.3}]\nmu2: {:.3} [{:.3}, {:.3}]\ngamma1: {:.3} [{:.3}, {:.3}]\ngamma2: {:.3} [{:.3}, {:.3}]", s, s_5, s_95, tau, tau_5, tau_95, mu1, mu1_5, mu1_95, mu2, mu2_5, mu2_95, gamma1, gamma1_5, gamma1_95, gamma2, gamma2_5, gamma2_95)

    }
}