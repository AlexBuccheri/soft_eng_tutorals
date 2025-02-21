# Software Engineering Tutorials

**NB**: Very much a WIP!

## Aims

Use fundamental algorithm/s found in electronic structure to demonstrate 
separation of implementation from algorithmic design. 

Implement in fortran, python and C++:

**fortran**

* Demonstrate limited functional injection
* Show abstraction with inheritance 
* Ultimately would be nice to test a plugin implementation

**python**

* Demonstrate functional injection
* Show abstraction with inheritance 
* Show abstraction with a plugin design

**C++** 

* Demonstrate functional injection: Lambdas and `std::func`
* Show abstraction with inheritance 
* Namespace-dependent look-up with templating
* Ultimately would be nice to test a plugin implementation


### Current Algorithms (in python)
* Linear conjugate gradient, including with basic preconditioning
* A less contrived example using non-linear CG, and the extension to BFGS.

**Of Note**

* Includes several test functions for optimisation, which have been scrubbed from wikipedia
  using the latest GPT model
* For the classic Rosenbrock function, I've coded the analytic derivative
* Computing the numerical derivative with central difference has also been implemented, to
  validate analytic derivatives

### TODOs

- [ ] Document the implementations more thoroughly.

**python**
- [ ] Abstract non-linear CG and BFGS, such that one can write the generic algorithm once,
then inject the implementation/method
   - Inject all the functions as arguments
   - Plugin design

**C++** 

- [ ] Add a build system, plus catch
- [ ] Implement and test non-linear CG
- [ ] Implement and test BFGS using a) armadillo and b) STL
- [ ] Test [nanobind](https://github.com/wjakob/nanobind) and expose C++ to python
- [ ] Abstract non-linear CG and BFGS, such that one can write the generic algorithm once,
  then inject the implementation/method
    - Inject all the functions as arguments
    - Class design
    - Templated algorithm

**fortran**

- [ ] Implement and test non-linear CG
- [ ] Abstract non-linear CG and BFGS, such that one can write the generic algorithm once,
  then inject the implementation/method
    - Inject some of the functions as arguments
    - Class design
