fn hello1(a: &mut Vec<i32>) {
    println!("Hello-1");
    a[0] = 10;
    println!("{:?}", a);
}

fn test1() {
    let mut a = vec![1,2,3];
    println!("{:?}", a);
    hello1(&mut a);
}

fn test2() {
    let mut a = vec![1,2,3];
    println!("{:?}", a);
    hello1(&mut a);
}

fn main() {
    test1();
    test2();
}