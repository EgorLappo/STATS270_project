use crate::data::{Data, Row};
use rand::prelude::*;
use rand::distributions::Distribution;
use statrs::distribution::{Uniform, Normal, Exp};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Parameters {
    s: f64,
    tau: f64,
    mu1: f64,
    mu2: f64,
    gamma1: f64,
    gamma2: f64,
}

pub fn run(data: Data, niter: usize, seed: usize) -> Parameters {
    // inits
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut samples: Vec<Parameters> = Vec::with_capacity(niter);
    let mut weights: Vec<f64> = Vec::with_capacity(niter);

    // for each iter 
    for _ in 0..niter {
        // propose
        let p = generate_sample(&mut rng);

        // compute weight
        let w = likelihood(&data, &p)/trial_likelihood(&p);

        // save to arrays
        samples.push(p);
        weights.push(w);
    }

    let wsum: f64 = weights.iter().sum();
    
    let sigma_mean: f64 = samples.iter().zip(weights.iter()).map(|(p, w)| p.s*w).sum::<f64>()/wsum;
    let tau_mean: f64 = samples.iter().zip(weights.iter()).map(|(p, w)| p.tau*w).sum::<f64>()/wsum;
    let mu1_mean: f64 = samples.iter().zip(weights.iter()).map(|(p, w)| p.mu1*w).sum::<f64>()/wsum;
    let mu2_mean: f64 = samples.iter().zip(weights.iter()).map(|(p, w)| p.mu2*w).sum::<f64>()/wsum;
    let gamma1_mean: f64 = samples.iter().zip(weights.iter()).map(|(p, w)| p.gamma1*w).sum::<f64>()/wsum;
    let gamma2_mean: f64 = samples.iter().zip(weights.iter()).map(|(p, w)| p.gamma2*w).sum::<f64>()/wsum;

    // save and print

    Parameters {
        s: sigma_mean,
        tau: tau_mean,
        mu1: mu1_mean,
        mu2: mu2_mean,
        gamma1: gamma1_mean,
        gamma2: gamma2_mean,
    }
}

fn generate_sample(rng: &mut StdRng) -> Parameters {
    let tau = Uniform::new(0.0, 1.0).unwrap().sample(rng);
    let s = Exp::new(12.0).unwrap().sample(rng);
    let mu1 = Normal::new(-1.5, 1.5).unwrap().sample(rng);
    let mu2 = Normal::new(-0.5, 1.5).unwrap().sample(rng);
    let gamma1 = Normal::new(-0.3, 1.5).unwrap().sample(rng);
    let gamma2 = Normal::new(0.3, 1.5).unwrap().sample(rng);

    Parameters {
        s,
        tau,
        mu1,
        mu2,
        gamma1,
        gamma2,
    }
}

fn likelihood(data: &Data, p: &Parameters) -> f64 {
    data.iter().map(|row| row_likelihood(&row, p)).product()
}

fn trial_likelihood(p: &Parameters) -> f64 {
    pnorm(p.mu1, -1.5,2.25)*pnorm(p.mu2, -0.5,2.25 )*pnorm(p.gamma1, -0.3, 2.25)*pnorm(p.gamma2, 0.3, 2.25)*pexp(p.s, 12.0)
}

fn row_likelihood(r: &Row, p: &Parameters) -> f64 {
    match r.group {
        1 => { pnorm(r.x1, p.mu1, p.s) * pnorm(r.x2, p.mu2, p.s)/p.s },
        2 => { pnorm(r.x1, p.gamma1, p.s) * pnorm(r.x2, p.gamma2, p.s)/p.s   },
        3 => { pnorm(r.x1, 0.5*p.mu1 + 0.5*p.gamma1, p.s) * pnorm(r.x2, 0.5*p.mu2 + 0.5*p.gamma2, p.s)/p.s  },
        4 => { pnorm(r.x1, p.tau*p.mu1 + (1. - p.tau)*p.gamma1, p.s) * pnorm(r.x2, p.tau*p.mu2 + (1. - p.tau)*p.gamma2, p.s)/p.s  },
        _ => unreachable!(),
    }
}

// normal likelihood  
fn pnorm(x:f64, mu: f64, s: f64) -> f64 {
    (-(x - mu).powi(2)/2.0/s).exp()/s.sqrt()
}

// exponential likelihood
fn pexp(x: f64, s: f64) -> f64 {
    (-x/s).exp()/s
}

impl Parameters {
    pub fn print_values(&self) {
        println!("s: {}", self.s);
        println!("tau: {}", self.tau);
        println!("mu1: {}", self.mu1);
        println!("mu2: {}", self.mu2);
        println!("gamma1: {}", self.gamma1);
        println!("gamma2: {}", self.gamma2);
    }
}