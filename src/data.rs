use csv::Reader;
use serde::Deserialize;
use color_eyre::Result;

#[derive(Deserialize, Debug)]
pub struct Row {
    pub group: u8,
    pub x1: f64,
    pub x2: f64
}

pub type Data = Vec<Row>;

pub fn load_data() -> Result<Data> {
    let mut data = Vec::new();
    let mut rdr = Reader::from_path("data.csv")?;
    for result in rdr.deserialize() {
        let row: Row = result?;
        data.push(row);
    }
    Ok(data)
}