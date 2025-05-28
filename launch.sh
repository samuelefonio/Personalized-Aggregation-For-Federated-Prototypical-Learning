#!/bin/sh

fluke federation config/exp.yaml config/fedproto.yaml

fluke federation config/exp.yaml config/fedhp.yaml

fluke sweep config/exp.yaml config/fedprotoIFCA.yaml

fluke sweep config/exp.yaml config/fedhpIFCA.yaml