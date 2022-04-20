#!/bin/bash

# Create data directory
mkdir -p data/

# Create obj.names
echo "butterfly" > data/obj.names

# Create obj.data
echo "classes = 1" > data/obj.data
echo "names = $PWD/data/obj.names" >> data/obj.data

