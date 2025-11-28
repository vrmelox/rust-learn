fn verdict_grimoires(livre: (&str, u16, bool)) {
    if livre.1 >= 500 && livre.2 == true {
        println!("rare");
    } else {
        println!("para");
    }
}

fn main() {
    let houbi = ("Houbilim", 275, true);
    let notre_dame = ("Notre-Dâme de Paris", 574, true);
    let west_chou = ("Western Tchoukoutou", 162, false);

    verdict_grimoires(houbi);
    verdict_grimoires(notre_dame);
    verdict_grimoires(west_chou);
}