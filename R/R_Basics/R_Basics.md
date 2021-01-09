---
layout: page
title: 'R: Basics'
permalink: /R/R-Basics/
---

# R Basics 


## 1. Running R Environments

For R programming, you can choose either a local or cloud environment. If you prefer to use a cloud environment, go to [RStudio Cloud](http://rstudio.cloud/). To run R on a local environment, you need to install [R](https://cran.r-project.org/) and [RStudio](https://rstudio.com/products/rstudio/download/) respectively. 

R is a programming language for statistics, and is one of the most popular computer languages. R is easy to learn and has been widely used by data scientists, social scientists, and digital humanists.

![RStudio](https://elibooklover.github.io/Tutorials/R/R_Basics/RBasics_01.png)
- Figure 1: RStudio

Figure 1 is a screenshot of RStudio. RStudio comprises of four spaces: `code editor`, `R console`, `workspace & history`, and `plots & files`. In `code editor`, you can load or save code. The file extension of the code notebook is `.R`. You can execute code with `cmd`+`return` (macOS) or `ctrl`+`enter` (Windows). If you want to stop running code, use `ctrl`+`c`.

Values or loaded data are saved in `workspace` on the top right. You can delete them with the `rm` function. You can see the history of executed code in the `History` tab. 

All code execution happens in `R console`. You can manage files in the `Files` tab on the bottom right. All visualizations are printed in the `Plots` tab. You can export visualizations through either code execution or the `Export` button in the `Plots` tab.

## 2. R Operations

In R, there are two types of operators: arithmetic and logical operators. Arithmetic operators are `+` (addition), `-` (subtraction), `*` (multiplication), `/` (division), and `^` / `**` (exponentiation). Logical operators are `>` (greater than), `>=` (greater than or equal to), `==` (equal to), and `!=` (not equal to).

R uses data structures such as scalars, vectors (numerical, character, and logical), matrices, data frames, and lists.

### 2.1 Scalar

A `scalar` is a single value. A `scalar` can have three data types: numeric, character, or logical. 


```R
x = 10
y = 3
```


```R
print(x)
print(y)
```

    [1] 10
    [1] 3
    


```R
print(c(x,y))
```

    [1] 10  3
    


```R
z <- x * 3 
```


```R
print(z)
```

    [1] 30
    


```R
x <- 1.2
y <- "hello"
z <- TRUE
```


```R
print(x)
print(y)
print(z)
```

    [1] 1.2
    [1] "hello"
    [1] TRUE
    

### 2.2 Vector

Vectors are a combination of numeric, character, or logical values. All elements must have the same mode (numeric, character , or logical).


```R
x <- c(1.1, 2.2, -5, 4.2, 2)
y <- c(TRUE, FALSE, TRUE)
z <- c("Howdy", "Aggies", "Whoop")
q <- 1.2:7.9 
```


```R
print(x)
print(y)
print(z)
print(q)
```

    [1]  1.1  2.2 -5.0  4.2  2.0
    [1]  TRUE FALSE  TRUE
    [1] "Howdy"  "Aggies" "Whoop" 
    [1] 1.2 2.2 3.2 4.2 5.2 6.2 7.2
    

### 2.3 Matrix

A matrix consists of rows and columns. All columns in a matrix must have the same type (numeric, character , or logical). A matrix is a homogenous collection of datasets.


```R
x <- matrix(1:20, nrow=5,ncol=4)
```


```R
print(x)
```

         [,1] [,2] [,3] [,4]
    [1,]    1    6   11   16
    [2,]    2    7   12   17
    [3,]    3    8   13   18
    [4,]    4    9   14   19
    [5,]    5   10   15   20
    


```R
print(x[3,]) # 3rd row of matrix
print(x[,2]) # 2nd column of matrix
```

    [1]  3  8 13 18
    [1]  6  7  8  9 10
    


```R
print(x[2:3,1:3]) # rows 2,3 of columns 1,2,3
```

         [,1] [,2] [,3]
    [1,]    2    7   12
    [2,]    3    8   13
    


```R
cells <- c(1812, 1819, 1870, 1880)
y <- matrix(cells, nrow=2, ncol=2, byrow=TRUE)
```


```R
print(y)
```

         [,1] [,2]
    [1,] 1812 1819
    [2,] 1870 1880
    


```R
cells <- c(1812, 1819, 1870, 1880)
rnames <- c("Born", "Death")
cnames <- c("Charles Dickens", "George Eliot")
z <- matrix(cells, nrow=2, ncol=2, byrow=TRUE, dimnames=list(rnames, cnames))
```


```R
print(z)
```

          Charles Dickens George Eliot
    Born             1812         1819
    Death            1870         1880
    

### 2.4 List

A `list` is a collection of elements. A `list` can include different types.


```R
x <- list(name="Charles Dickens", gender="M", nationality="English", born=1812, matrix_example=z)
```


```R
print(x)
```

    $name
    [1] "Charles Dickens"
    
    $gender
    [1] "M"
    
    $nationality
    [1] "English"
    
    $born
    [1] 1812
    
    $matrix_example
          Charles Dickens George Eliot
    Born             1812         1819
    Death            1870         1880
    
    

### 2.5 Data Frame

A data frame can contain different data types. Data frames are mostly used for storing data. A data frame is similar to a table in Excel. The data mode must be numeric, character or logical. A data frame is a heterogeneous collection of datasets.


```R
df <- data.frame(num=c(1:3), author=c("Charles Dickens", "George Eliot", "Wilkie Collins"), birth_date=as.Date(c("1812/2/7", "1819/11/22", "1824/1/8")), death_year = as.Date(c("1870/6/9", "1880/12/22", "1889/9/23")), children=c(10, 0, 3)) 
```


```R
print(df)
```

      num          author birth_date death_year children
    1   1 Charles Dickens 1812-02-07 1870-06-09       10
    2   2    George Eliot 1819-11-22 1880-12-22        0
    3   3  Wilkie Collins 1824-01-08 1889-09-23        3
    
