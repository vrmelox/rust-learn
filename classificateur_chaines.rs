fn classifier_chaine(matiere: &str) -> &str {
    match matiere {
        "argent" => {"Médecine"},
        "or" => {"Economie"},
        "acier" => {"Guerre"},
        _ => {"Matière invalide."},
    }
}

fn main() {
    println!("Selon votre matière, votre domaine d'expertise est {}", classifier_chaine("or"))
}