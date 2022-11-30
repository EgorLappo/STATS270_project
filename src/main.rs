use color_eyre::Result;

pub mod data;
pub mod mh;
pub mod hmc;

fn main() -> Result<()>{
    let data = data::load_data()?;

    let mut hmc_chain = hmc::Chain::new(data);

    let hmc_samples = hmc_chain.run(1000, 2000, 42);

    println!("{}", hmc::Parameters::summary(hmc_samples));

    Ok(())
}
