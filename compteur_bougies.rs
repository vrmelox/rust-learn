fn calculer_bougies(salles: u32, heures: u32) -> u32 {
    let total_heures = 5 * heures;
    salles * total_heures
}

fn main() {
    println!("Pour {} heures, les mestres doivent utiliser {} bougies.", 8, calculer_bougies(12, 8));
}