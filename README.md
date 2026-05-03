# CuMallat

This repo contains my implementation of Mallat's algorithm for DTWT (Discrete Time Wavelet Transform) using Cuda.

The idea came after I had a class with my teacher [Rodrigo Capobianco Guido](https://scholar.google.com/citations?user=Jx8uPsEAAAAJ&hl=en) on the topic during my Master's in Computer Science at UNESP. During the class we learned the method and I thought that it would be amazing to do using Cuda capabilities.

So I started doing when I had some time, and here we are. My idea was not to create something with vast abilities, options and the best software for processing such signals, but actually learn more about the method and cuda in general, also to have fun while learning. Saying so, Yes, you'll may face a lot of bugs and some wrong results while using. 

I did some tests which you can find at: [tests.cu](./tests.cu). At first, I tried many different tools for this, but it was too complex for this quick project, so I done a test suite myself using pure c++.

The processing functions themselves are stored at: [mallat.hpp](./mallat.hpp), which is a cuda library you can include in your code. There're plenty of room for improvement. In case you find anything you'll modify to get better performance or correctness, feel free to open a pull request!

The Code is able to calculate the DTWT and its inverse. Accepting only even sized filters and signals, I'm not sure if there're odd sized ones in the wild, but since these are the ones we used during our classes, I decided to limit my code this way. On the other hand, as far as I tested, the code is also able to handle wrap around and scenarios which the filter is larger than the signal.

There's a sample code at: [example.cu](./example.cu) which you can run a simple scenario with Haar Filters to understand it better. For those who doesn't have a CUDA device, I let a small jupyter notebook which you can run on google colab laveraging the T4 GPU freely available. Make sure to create a session with GPU.

<a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/github/Dpbm/cumallat/blob/main/colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


Finally, There're two commands you can run to build this project:

```bash
make //for building the 'production' code
make debug //for debugging and run tests
```
