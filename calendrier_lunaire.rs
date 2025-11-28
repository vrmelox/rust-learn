fn main() {
    let mut count = 1;
    while count <= 28 {
        if count == 14 {
            println!("Jour {} : Pleine lune.", count);
        } else if count == 1 || count == 28 {
            println!("Jour {} : Nouvelle lune.", count);
        } else {
            println!("Jour {} : Phase lunaire.", count);
        }
        count += 1;
    }
}