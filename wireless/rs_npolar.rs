use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let n_iter = 1;
    let k_code = 2;
    // u_array = np.random.randint(2, size=(N_iter, K_code))
    let u_array: Vec<u32> = (0..n_iter).map(|_| rng.gen_range(0,1)).collect();
    println!("{:?}", u_array);
}