fn min_max_moy(tabs: &[u16;10]) -> (u16, u16, u16) {
    let mini = tabs.iter().min().unwrap();
    let maxi = tabs.iter().max().unwrap();
    let moy = tabs.iter().map(|&x| x).sum::<u16>() / tabs.len() as u16;
    return (*mini, *maxi, moy);
}


fn main() {
    let tabs = [37, 84, 12, 59, 6, 93, 41, 27, 70, 18];
    let ser = min_max_moy(&tabs);
    println!("Le tabs donne {} comme min; {} comme max et {} comme moyenne", ser.0, ser.1, ser.2);
}