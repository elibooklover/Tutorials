---
layout: page
title: 'Python: pandas and NumPy'
permalink: /Python/pandas/
---

# Introduction to pandas & NumPy

`pandas` is a popular data analysis library in Python. You can think of `pandas` as a back-end Excel tool that can be customized in order to deal with raw data more easily. `pandas` deals with 1D and 2D (dimensional) arrays.  

## 1. NumPy


```python
import numpy as np
```


```python
np.array([1,3,5])
```




    array([1, 3, 5])




```python
np.array([[1,3,5], [2,4,6]])
```




    array([[1, 3, 5],
           [2, 4, 6]])




```python
np.zeros([3,6])
```




    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])




```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
a.shape
```




    (3, 4)




```python
a = np.array(['a', 'b', 3, 4])
```


```python
print(a[1:3])
```

    ['b' '3']



```python
b = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(b)
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]



```python
print(b[0:2, 1:3])
```

    [[2 3]
     [6 7]]



```python
print(b[:, 1:3])
```

    [[ 2  3]
     [ 6  7]
     [10 11]]



```python
print(b[1:, :])
```

    [[ 5  6  7  8]
     [ 9 10 11 12]]



```python
print(b[:, :])
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]


Let's use `NumPy` for calculation.


```python
n1 = np.array([3, 6, 9])

print(n1)
print(n1 + 5)
print(n1 - 5)
print(n1 * 5)
print(n1 / 5)
```

    [3 6 9]
    [ 8 11 14]
    [-2  1  4]
    [15 30 45]
    [0.6 1.2 1.8]
    [3 1 4]



```python
n2 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(np.max(n2))
print(np.min(n2))
print(np.mean(n2))
print(np.median(n2))
print(np.var(n2))
print(np.std(n2))
```

    10
    1
    5.5
    5.5
    8.25
    2.8722813232690143


## 2. pandas

`pandas` has two data structures: `Series` for 1D data and `DataFrame` for 2D data.


```python
# Series

s1 = pd.Series({'Boffin': 100, 'Bella': 90, 'John': 85})
s1.name = 'Literature'
print(s1)
```

    Boffin    100
    Bella      90
    John       85
    Name: Literature, dtype: int64



```python
# DataFrame

pd.DataFrame({
    'age':[10, 42, 21]
})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
{
    'age':[10, 42, 21]
}
```




    {'age': [10, 42, 21]}




```python
{
  'age':[10, 42, 10],
  'name': ['Boffin', 'Bella', 'John'],
  'gender': ['male', 'female', 'male']
}
```




    {'age': [10, 42, 10],
     'gender': ['male', 'female', 'male'],
     'name': ['Boffin', 'Bella', 'John']}




```python
pd.DataFrame({
  'age':[10, 42, 21]
}).join(
pd.DataFrame({
  'name': ['Boffin', 'Bella', 'John']
})
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>Boffin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>Bella</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>John</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({
  'age':[10, 42, 21]
  }).join(
      pd.DataFrame({
        'name': ['Boffin', 'Bella', 'John']
        })
    ).join(
      pd.DataFrame({
        'city': ['Bryan', 'Seoul', 'London']
      })
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>name</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>Boffin</td>
      <td>Bryan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>Bella</td>
      <td>Seoul</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>John</td>
      <td>London</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({
  'age':[10, 42, 21]
}).append(
pd.DataFrame({
  'name': ['Boffin', 'Bella', 'John']
})
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Boffin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Bella</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>John</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({
  'age':[10, 42, 21],
  'name': ['Boffin', 'Bella', 'John']
}).merge(
pd.DataFrame({
  'name': ['Boffin', 'Bella', 'John'],
  'city': ['Bryan', 'Seoul', 'London']
}), on=['name']
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>name</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>Boffin</td>
      <td>Bryan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>Bella</td>
      <td>Seoul</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>John</td>
      <td>London</td>
    </tr>
  </tbody>
</table>
</div>


