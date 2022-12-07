use crate::data::Data;
use rand::prelude::*;
use rand::distributions::Distribution;
use statrs::distribution::Normal;
use serde::{Serialize, Deserialize};

static SPROPSD: f64 = 0.1;
static MEANPROPSD: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct Parameters {
    L: usize,
    dt: f64,
    m: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutParameters {
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
            L: 5,
            dt: 0.001,
            m: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            p: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            q: vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
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
        let (q_new, p_new) = self.leapfrog_propose(rng);

        let alpha: f64 = expH(&self.data, &self.parameters.m, &q_new, &p_new) / expH(&self.data, &self.parameters.m, &self.parameters.q, &self.parameters.p);

        if rng.gen::<f64>() < alpha {
            self.parameters.q = q_new;
            self.parameters.p = p_new;
        }
    }

    fn leapfrog_propose(&mut self, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>) {
        // use the fact that the mass matrix is diagonal
        for i in 0..6 {
            self.parameters.p[i] = Normal::new(0.0, self.parameters.m[i]).unwrap().sample(rng);
        }

        let mut pn = self.parameters.p.clone();
        let mut qn = self.parameters.q.clone();

        for _ in 0..self.parameters.L {
            let du = dU(&self.data, &qn);

            for i in 0..6 {
                pn[i] -= 0.5 * self.parameters.dt * du[i];
                qn[i] += self.parameters.dt * self.parameters.m[i] * pn[i];
                pn[i] -= 0.5 * self.parameters.dt * du[i];
            }
        }

        (qn, pn)
    }
}

fn expH(data: &Data, m: &Vec<f64>, q: &Vec<f64>,  p: &Vec<f64>) -> f64 {
    let u = U(data, q);
    let mut v: f64 = 0.0;
    for (mi, pi) in m.iter().zip(p.iter()) {
        v += mi * pi * pi;
    }
    let H = - v / 2.0 - u;
    H.exp()
}

fn U(data: &Data, q: &Vec<f64>) -> f64 {
    let N = data.len() as f64;
    let mut u = N*(6.2831_f64.ln()) + (N+1.)*q[0].ln();

    for r in data.iter() {
        match r.group {
            1 => { 
                let t = (r.x1 - q[2]).powi(2) + (r.x2 - q[3]).powi(2); 
                u += t/(2.*q[0]);
            },
            2 => {
                let t = (r.x1 - q[4]).powi(2) + (r.x2 - q[5]).powi(2); 
                u += t/(2.*q[0]);
            },
            3 => {
                let t = (r.x1 - q[2]*0.5 - q[4]*0.5).powi(2) + (r.x2 - q[3]*0.5 - q[5]*0.5).powi(2); 
                u += t/(2.*q[0]);
            },
            4 => {
                let t = (r.x1 - q[2]*q[1] - q[4]*(1. - q[1])).powi(2) + (r.x2  - q[3]*q[1] - q[5]*(1. - q[1])).powi(2); 
                u += t/(2.*q[0]);
            },
            _ => unreachable!(),
        }
    }

    u
}

fn dU(data: &Data, q: &Vec<f64>) -> Vec<f64> {
    let N = data.len() as f64;

    let mut du = vec![0.; 6];

    du[0] += (N+1.)/q[0];

    for row in data.iter() {
        match row.group {
            1 => {
                let t0 = (row.x1 - q[2]).powi(2) + (row.x2 - q[3]).powi(2);
                du[0] -= t0/(2.*q[0].powi(2));

                du[2] -= (row.x1 - q[2])/q[0];
                du[3] -= (row.x2 - q[3])/q[0];
            },
            2 => {
                let t0 = (row.x1 - q[4]).powi(2) + (row.x2 - q[5]).powi(2);
                du[0] -= t0/(2.*q[0].powi(2));

                du[4] -= (row.x1 - q[4])/q[0];
                du[5] -= (row.x2 - q[5])/q[0];
            },
            3 => {
                let t0 = (row.x1 - q[2]*0.5 - q[4]*0.5).powi(2) + (row.x2 - q[3]*0.5 - q[5]*0.5).powi(2);
                du[0] -= t0/(2.*q[0].powi(2));

                let t2 = (row.x1-q[2]*0.5-q[4]*0.5)/(2.*q[0]);
                let t3 = (row.x2-q[3]*0.5-q[5]*0.5)/(2.*q[0]);
                du[2] -= t2;
                du[3] -= t3;
                du[4] -= t2;
                du[5] -= t3;

            },
            4 => {
                let t0 = (row.x1 - q[2]*q[1] - q[4]*(1. - q[1])).powi(2) + (row.x2 - q[3]*q[1] - q[5]*(1. - q[1])).powi(2);
                du[0] -= t0/(2.*q[0].powi(2));

                let t1 = (row.x1 - q[2]*q[1] - q[4]*(1. - q[1]))*(q[4] - q[2]) + (row.x2 - q[3]*q[1] - q[5]*(1. - q[1]))*(q[5] - q[3]); 
                du[1] += t1/q[0];
                
                let t2 = (row.x1 - q[2]*q[1] - q[4]*(1.-q[1]))/q[0];
                let t3 = (row.x2 - q[3]*q[1] - q[5]*(1.-q[1]))/q[0];
                du[2] -= q[1]*t2;
                du[3] -= q[1]*t3;
                du[4] -= (1.-q[1])*t2;
                du[5] -= (1.-q[1])*t3;

            },
            _ => unreachable!(),
        }
    }

    du
}


impl OutParameters {
    pub fn from_parameters(p: &Parameters) -> OutParameters {
        OutParameters {
            s: p.s(),
            tau: p.tau(),
            mu1: p.mu1(),
            mu2: p.mu2(),
            gamma1: p.gamma1(),
            gamma2: p.gamma2(),

        }
    }

    pub fn save_to_csv(ps: &Vec<Parameters>, filename: &str) {
        let mut wtr = csv::Writer::from_path(filename).unwrap();

        for p in ps.iter().map(|p| OutParameters::from_parameters(p)) {
            wtr.serialize(p).unwrap();
        }
    }
}

impl Parameters {
    fn s(&self) -> f64 {
        self.q[0]
    }

    fn tau(&self) -> f64 {
        self.q[1]
    }

    fn mu1(&self) -> f64 {
        self.q[2]
    }

    fn mu2(&self) -> f64 {
        self.q[3]
    }

    fn gamma1(&self) -> f64 {
        self.q[4]
    }

    fn gamma2(&self) -> f64 {
        self.q[5]
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
            s += p.s();
            tau += p.tau();
            mu1 += p.mu1();
            mu2 += p.mu2();
            gamma1 += p.gamma1();
            gamma2 += p.gamma2();
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
            s_vec.push(p.s());
            tau_vec.push(p.tau());
            mu1_vec.push(p.mu1());
            mu2_vec.push(p.mu2());
            gamma1_vec.push(p.gamma1());
            gamma2_vec.push(p.gamma2());
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