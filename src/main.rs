use color_eyre::Result;

pub mod data;

fn main() -> Result<()>{
    let data = data::load_data()?;
    println!("{:?}", data);

    Ok(())
}
