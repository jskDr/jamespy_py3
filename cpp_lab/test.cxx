// C++ program to illustrate the
// iterators in vector
#include <iostream>
#include <vector>
  
using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    // vector<int> results;
    int i, j, L = nums.size(), new_target = 0;
    for(i = 0; i < L; i++)
        // new_target = target - nums[i];
        cout << "Computing result:" << endl;
        cout << target << endl;
        cout << nums[i] <<endl;
        cout << new_target << endl;
        for(j = i+1; j < L; j++)
            if (nums[j] == new_target) {
                // results.push_back(i);
                // results.push_back(j);
                nums.clear();
                nums.push_back(i);
                nums.push_back(j);
                return nums;
            }
    // return results;
    return nums;
}
  
int main()
{
    vector<int> nums;
    nums.push_back(2);
    nums.push_back(7);
    nums.push_back(11);
    nums.push_back(15);
    for(auto i = nums.begin(); i != nums.end(); ++i)
        cout << *i << endl;

    auto r = twoSum( nums, 9); 
    for(int i = 1; i != r.size(); ++i) 
        cout << i << " ";
    cout << endl;
       
    return 0;
}
