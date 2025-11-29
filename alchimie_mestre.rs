use std::io;

fn afficher_recette(recettes: &[(&str, Vec<&str>, u16); 5]) {
    println!("\n--- Toutes les recettes ---");
    for recette in recettes {
        println!("• {} - {} min", recette.0, recette.2);
        println!("  Ingrédients: {}", recette.1.join(", "));
    }
}

fn recettes_rapides(recettes: &[(&str, Vec<&str>, u16); 5], temps_max: u16) {
    println!("\n--- Recettes rapides (≤ {} min) ---", temps_max);
    let mut trouvees = false;
    for recette in recettes {
        if recette.2 <= temps_max {
            println!("• {} - {} min", recette.0, recette.2);
            trouvees = true;
        }
    }
    if !trouvees {
        println!("Aucune recette trouvée.");
    }
}

fn calculer_temps_total(recettes: &[(&str, Vec<&str>, u16); 5]) {
    let total: u16 = recettes.iter().map(|r| r.2).sum();
    println!("\n--- Temps total ---");
    println!("Le temps de cuisson total est de : {} minutes.", total);
}

fn main() {
    let recettes: [(&str, Vec<&str>, u16); 5] = [
        ("Salade César", vec!["Laitue romaine", "Poulet grillé", "Croutons", "Parmesan", "Sauce césar"], 20),
        ("Ratatouille", vec!["Aubergines", "Courgettes", "Poivrons", "Tomates", "Oignons", "Ail", "Herbes de Provence"], 30),
        ("Quiche Lorraine", vec!["Pâte brisée", "Œufs", "Crème fraîche", "Lardons", "Fromage râpé"], 45),
        ("Curry de lentilles", vec!["Lentilles", "Lait de coco", "Épinards", "Oignon", "Ail", "Curry en poudre"], 35),
        ("Poulet au miel", vec!["Cuisse de poulet", "Miel", "Sauce soja", "Ail", "Gingembre"], 20)
    ];

    loop {
        println!("\n=========================================================");
        println!("Bienvenue sur les recettes du Gustomberi.");
        println!("1. Afficher toutes les recettes");
        println!("2. Recettes rapides");
        println!("3. Calculer le temps total");
        println!("0. Quitter");
        println!("=========================================================");
        print!("Entrez votre choix : ");
//        io::Write::flush(&mut io::stdout()).unwrap();
        
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Erreur de lecture");

        let choix: u16 = match input.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("❌ Erreur: Entrez un nombre valide.");
                continue;
            }
        };

        match choix {
            0 => {
                println!("\nAu revoir ! 👋");
                break;
            }
            1 => afficher_recette(&recettes),
            2 => recettes_rapides(&recettes, 25),
            3 => calculer_temps_total(&recettes),
            _ => println!("❌ Choix erroné. Veuillez choisir entre 0 et 3."), 
        }
    }
}