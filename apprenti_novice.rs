fn main () {
    let nom: String = String::from("Sorto");
    let mut age: u8 = 17;
    const ANNEE_FONDATION_CITADELLE: u16 = 1850;
    let maison_origine: String = String::from("Oldtwon");
    println!("L'apprenti {}, originaire de {}, a {} ans.", nom, maison_origine, age);
    age = 20;
    println!("L'apprenti {}, originaire de {}, a {} ans. Il était ici depuis 3 ans.", nom, maison_origine, age);
    println!("La citadelle d'Oldtown a été créé en {}.", ANNEE_FONDATION_CITADELLE);
}