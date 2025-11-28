fn tableau_honneur(scores: [u8; 7]) {
    let mut total: u16 = 0;
    let mut reussis = 0;
    for i in scores {
        total += i as u16;
        if i >= 80 {
            reussis += 1;
        }
    }
    let moy = total / 7;
    println!("La moyenne totale est de {}.", moy);
    let maxi = scores.iter().max().unwrap();
    println!("Le score le plus élevé est de {}.", *maxi);
    println!("{} novices ont réussi leur examens.", reussis);
}

fn main() {
    let scores = [85,92,78,95,88,90,73];
    tableau_honneur(scores);
}