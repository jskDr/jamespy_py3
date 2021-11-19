extern crate rand;
use rand::distributions::{Distribution, Uniform};

fn main() {
    let between = Uniform::from(10..10000);
    let mut rng = rand::thread_rng();
    let mut sum = 0;
    for _ in 0..1000 {
        sum += between.sample(&mut rng);
    }
    println!("{}", sum);
}