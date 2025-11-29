use std::io

fn afficher_recette(recettes: [(&str, Vec<&str>, u16);5]) {
    for recette in recettes {
        println!("{:?}.", recette);
    }
}

fn recettes_rapides(recettes: [(&str, Vec<&str>, u16);5], temps_max: u32) {
    for recette in recettes {
        if recette.2 as u32 == temps_max {
            println!("{:?}", recette);
        }
    }
}

fn calculer_temps_total(recettes: [(&str, Vec<&str>, u16);5]) {
    let total: u16 = 0;
    for recette in recettes {
        total += recette.2;
    }
    println!("Le temps de cuisson total est de : {}.", total);
}

fn main() {
    // let recettes: Vec<(&str, Vec<&str>, u16)> = vec!
    let recettes: [(&str, Vec<&str>, u16); 5] = [
        ("Salade César", vec!["Laitus romaine", "Poulet grillé", "Croutons", "Parmesan", "Sauce césar"], 20),
        ("Ratatouille", vec!["Aubergines", "Courgettes", "Poivrons", "Tomates", "Oignons", "Ail", "Herbes de provence"], 30),
        ("Quiche Lorraine", vec!["Pâte brisée", "Oeufs", "Crème fraîche", "Lardons", "Fromage râpé"], 45),
        ("Curry de lentilles", vec!["Lentilles", "Lait de coco", "Epinards", "Oignon", "Ail", "Curry en poudre"], 35),
        ("Poulet en miel", vec!["Cuisse de poulet", "Miel", "Lait de soja", "Ail", "Gingembre"], 20)
    ];
    println!("Bienvenue sur les recettes du Gustomberi.");
    println!("1.Afficher toutes les recettes.\n2.Recettes rapides.\n3.Calculer le temps total.")
    let mut input = String::new();
    println!("Entrez votre choix :");
    io::stdin()
        .read_line(&mut input)
        .expect("Erreur de lecture");

    let choix: u16 = input.trim().parse().expect("Entrez un nombre valide.");

    match choix {
        1 => afficher_recette(recettes),
        2 => recettes_rapides(recettes, 25),
        2 => calculer_temps_total(recettes),
        _= afficher_recette(recettes), 
    }
}