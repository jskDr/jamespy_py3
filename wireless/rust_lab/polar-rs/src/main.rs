use rand::{distributions::Uniform, Rng};

fn main() {
    let mut rng = rand::thread_rng();

    let k_code = 2;

    // u_array = np.random.randint(2, size=(N_iter, K_code))
    let range = Uniform::new(0, 2);
    let u_array: Vec<u32> = (0..k_code).map(|_| rng.sample(&range)).collect();

    println!("{:?}", u_array);
}