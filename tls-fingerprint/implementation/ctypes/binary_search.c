int binary_search(int* array, int length, int x)
{
    int first = 0;
    int last = length - 1;
    int mid = 0;

    while(first <= last)
    {
        mid = (int)((first + last) / 2);

        if(array[mid] < x)
        {
            first = mid + 1;
        }
        else if(array[mid] > x)
        {
            last = mid - 1;
        }
        else
        {
            return mid;
        }
            
    }

    return -1;
}