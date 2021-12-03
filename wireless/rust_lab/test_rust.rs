fn sort_vec_usize() {
    let mut data: Vec<usize> = vec![1, 5, 2, 8, 3];
    println!("Origital data is given by data={:?}", data);
    data.sort();
    println!("By sorting, the order of the elements in the data is changed and the result is given by sorted data={:?}", data)
}

fn sort_vec_f32() {
    let mut data: Vec<f32> = vec![1.1, 5.2, 2.8, 8.3, 3.1];
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{:?}", data);
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Person {
    name: String,
    age: u32
}

impl Person {
    pub fn new(name:String, age: u32) -> Self {
        Person {name, age}
    }
}

fn sort_struct() {
    let mut people = vec![
        Person::new("Zoe".to_string(), 25),
        Person::new("Al".to_string(), 60),
        Person::new("John".to_string(), 1),
    ];    
    println!("{}", people[0].age);
    people[1].age = 61; // From this year, Al's age becomes 61
    println!{"{:?}", people[1]};
    println!("Origial people={:?}", people);
    people.sort();
    println!("Sorted people={:?}", people);
    people.sort_by(|a, b| b.age.cmp(&a.age));
    println!("Age sorted people={:?}", people);
}

fn main() {
    sort_vec_usize();
    sort_vec_f32();
    sort_struct();
}