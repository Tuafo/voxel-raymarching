use std::{collections::HashMap, error::Error, io::Read, u8};

pub trait PropertiesExt {
    fn parse_or<T: std::str::FromStr>(&self, name: &str, default: T) -> T;
}
impl PropertiesExt for HashMap<String, String> {
    fn parse_or<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        self.get(name)
            .and_then(|s| s.parse::<T>().ok())
            .unwrap_or(default)
    }
}

pub trait ReadExt {
    fn read_bytes<const N: usize>(&mut self) -> Result<[u8; N], Box<dyn Error>>;
    fn read_i32(&mut self) -> Result<i32, Box<dyn Error>>;
    fn read_vox_string(&mut self) -> Result<String, Box<dyn Error>>;
    fn read_vox_dict(&mut self) -> Result<HashMap<String, String>, Box<dyn Error>>;
}
impl<T: Read> ReadExt for T {
    fn read_bytes<const N: usize>(&mut self) -> Result<[u8; N], Box<dyn Error>> {
        let mut buf = [0u8; N];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn read_i32(&mut self) -> Result<i32, Box<dyn Error>> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_vox_string(&mut self) -> Result<String, Box<dyn Error>> {
        let length = self.read_i32()?;
        let mut buf = vec![0u8; length as usize];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf)?)
    }

    fn read_vox_dict(&mut self) -> Result<HashMap<String, String>, Box<dyn Error>> {
        let pair_count = self.read_i32()?;
        let mut res = HashMap::new();
        for _ in 0..pair_count {
            let k = self.read_vox_string()?;
            let v = self.read_vox_string()?;
            res.insert(k, v);
        }
        Ok(res)
    }
}
