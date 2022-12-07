use color_eyre::Result;

pub mod data;
pub mod mh;
pub mod hmc;
pub mod importance;

fn main() -> Result<()>{

    println!("loading data...");

    let data = data::load_data()?;

    println!("running Metropolis-Hastings...");

    let mut mh_chain = mh::Chain::new(data.clone());

    let mh_samples = mh_chain.run(1000, 3000, 42);

    println!("MH results:");

    println!("{}", mh::Parameters::summary(&mh_samples));

    println!("saving the samples to file 'mh_samples.csv'...");

    mh::Parameters::save_to_csv(&mh_samples, "mh_samples.csv");

    println!("running Hamiltonian Monte Carlo...");

    let mut hmc_chain = hmc::Chain::new(data.clone());

    let hmc_samples = hmc_chain.run(3000, 8000, 42);

    println!("HMC results:");

    println!("{}", hmc::Parameters::summary(&hmc_samples));

    println!("saving the samples to file 'hmc_samples.csv'...");

    hmc::OutParameters::save_to_csv(&hmc_samples, "hmc_samples.csv");

    println!("russing importance sampling...");

    let importance_samples = importance::run(data.clone(), 10000, 42);

    println!("importance sampling results:");

    importance_samples.print_values();

    println!("not yet running the Gibbs sampler...");

    Ok(())
}
