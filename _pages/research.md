---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---

# The Big Picture
In the broadest sense, my research seeks to explain why the Universe is ordered instead of being a chaotic mess. One of the best examples of this is galaxies: they had their origins as random quantum fluctuations in the early Universe, but today they resemble organized collections of stars, gas, dust, black holes, planets, etc. Galaxies come in all shapes, sizes and colors -- I use a combination of observations, theory and data science to understand how and why. It is important to decode the astrophysics of galaxies because they represent one of our most promising probes of fundamental physics via cosmology -- their numbers, motions and spatial clustering have historically provided a variety of evidence for dark matter, dark energy and inflation. This has motivated ambitious upcoming telescopes such as Roman, Rubin, Euclid and Simons which will map tens of billions of galaxies and their gaseous cosmic ecosystems. Without transforming galaxy formation into a precision science, as I am trying to do, it is going to be impossible to confidently interpret this data to constrain the origin, evolution and fate of our Universe, and answer one of the biggest questions of them all: how did we get here? 


# Understanding Galaxies as Dynamical Systems

During my [NSF-funded PhD at UC Santa Cruz](https://escholarship.org/uc/item/9xc1v7c9) and my NASA Hubble Postdoctoral Fellowship at Columbia University, I explored fundamentally new ways to evolve and understand galaxies as dynamical systems. My PhD thesis was the only one exclusively focused on the [Simulating Multi-scale Astrophysics to Understand Galaxies (SMAUG) Project](https://www.simonsfoundation.org/flatiron/center-for-computational-astrophysics/galaxy-formation/smaug/) to ever come out of that collaboration. As a core member of SMAUG, I was fortunate to analyze simulations from the [Feedback In Realistic Environments (FIRE) Project](https://fire.northwestern.edu/) as an initial testbed for my research, later generalizing to other simulations and datasets. As part of the [Simons Collaboration on Learning the Universe (LtU)](https://learning-the-universe.org/), I independently conceived and led the development of [```sapphire```](https://github.com/virajpandya/sapphire), which bridges galactic dynamics, numerics, statistics, astrophysics and cosmology in "next-generation" ways and is built on top of the [JAX Python Library](https://github.com/jax-ml/jax). 

My research in this area falls under three themes: 

## Hybrid physics-informed, data-driven galaxy simulators
One of the grand challenges of modern astrophysics is to develop a fully predictive theory of galaxy formation so that galaxies can become more robust cosmological probes. However, this requires overcoming five obstacles: (1) the relevant physical processes span a large dynamic range, (2) they are non-linearly coupled across scales leading to complicated emergent behavior, (3) many of those processes are not understood from first principles, (4) we cannot directly observe the evolution of individual galaxies on human timescales so must resort to statistical, population-level studies, and (5) we have noisy, incomplete data. Figure 1 of [Pandya+26](https://ui.adsabs.harvard.edu/abs/2026arXiv260406318P/abstract) provides a schematic overview of [```sapphire```](https://github.com/virajpandya/sapphire) and my new interdisciplinary dynamical systems vision for achieving a hybrid physics-informed, data-driven understanding of galaxy evolution. 


<img width="1117" height="581" alt="image" src="https://github.com/user-attachments/assets/e0af1c0c-003a-4b96-b0ed-5843e0991bbd" />


## Energy flows in galactic atmospheres
Like planets and stars, galaxies have atmospheres that regulate energy balance. During my PhD and Hubble Fellowship, I prototyped modeling energy flows between galaxies and their gaseous atmospheres because the thermodynamic state of the latter may hold new clues about both astrophysics and cosmology. Figure 1 of [Pandya+23](https://ui.adsabs.harvard.edu/abs/2023ApJ...956..118P/abstract) illustrates a new physical model in which eight non-linearly coupled ordinary differential equations describe how mass, metals and energy flow between galaxies and their gaseous atmospheres. This physical model was developed in collaboration with [Dr. Drummond Fielding](https://dfielding14.github.io/), [Dr. Greg Bryan](https://www.astro.columbia.edu/content/greg-bryan), [Dr. Rachel Somerville](https://www.simonsfoundation.org/people/rachel-somerville/) and [Dr. Chris Carr](https://dof.princeton.edu/people/chris-carr). We were guided by my earlier analysis of such flows in the "Santa Cruz" galaxy formation model originally created by Dr. Rachel Somerville as well as the FIRE simulations, following even earlier work by Dr. Drummond Fielding and [Dr. Daniel Anglés-Alcázar](https://physics.uconn.edu/person/daniel-angles-alcazar/) ([Pandya+20](https://ui.adsabs.harvard.edu/abs/2020ApJ...905....4P/abstract), [Pandya+21](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2979P/abstract)). With [Prof. Mark Voit](https://gmvoit.org/), we developed a generalized variant of this model called [ExpCGM](https://gmvoit.github.io/ExpCGM/). My students [Austen Gabrielpillai](https://aust427.github.io/) and [Bill Robinson](https://www.linkedin.com/in/bill-robinson-8b373110) are adding satellite galaxies to this picture and my collaborator [Dr. Bryan Terrazas](https://www.oberlin.edu/bryan-terrazas) is exploring black holes. With my student [Yossi Oren](https://ui.adsabs.harvard.edu/abs/2026ApJ...999..259O/abstract) and collaborator [Dr. Osase Omoruyi](https://web.astro.princeton.edu/people/osase-omoruyi), we validated and extended the approach from my PhD thesis to another set of simulations: [IllustrisTNG](https://www.tng-project.org/). Following my earlier work with [Dr. Yakov Faerman](https://astro.washington.edu/people/yakov-faerman), Dr. Rachel Somerville and [Dr. Amiel Sternberg](https://www.simonsfoundation.org/people/amiel-sternberg/), we will soon be in a position to forward model direct observables of galactic atmospheres.




<img width="949" height="680" alt="image" src="https://github.com/user-attachments/assets/2ddf777c-a7f8-436d-93b3-c5caea006c0b" />


## Differentiable, GPU-accelerated, interpretable galaxy evolution
I finished my PhD in 2021 before the hype of large language models. Ever since, I have been fascinated by automatic differentiation and GPU/TPU parallelization which, together with algorithmic advances, underpin the recent AI/ML revolution. Thanks to JAX and Julia, I have been exploring how to leverage these technological advancements to accelerate and improve the interpretability and causal identifiability of our models. Figure 4 of [Pandya+26](https://ui.adsabs.harvard.edu/abs/2026arXiv260406318P/abstract) shows a Jacobian matrix with interpretable, non-random structures that encode the astrophysical sensitivity and locally linearized dynamics of a model Milky Way-like galaxy. These gradients provide a fundamentally new way of understanding the behavior, successes and limitations of our models. They also unlock previously inaccessible techniques to perform very fast Bayesian inference for galaxy formation, as is standard in precision cosmology. With my collaborators [Dr. Lucas Makinen](https://tlmakinen.github.io/), [Dr. Matthew Ho](https://maho3.github.io/), [Dr. Kartheik Iyer](https://kartheikiyer.github.io/), [Dr. Chris Lovell](https://www.christopherlovell.co.uk/) and others, we are exploring implicit likelihood inference and synthetic observations.

<img width="963" height="573" alt="image" src="https://github.com/user-attachments/assets/8a1e5d14-2dc9-4946-af1a-015de9371373" />


# Galaxy Morphology, Kinematics and Scaling Relations

Galaxies are not a chaotic mess: they appear to follow morphological sequences and exhibit remarkably tight correlations both globally, locally and across time. These population-level trends provide essential information since the evolution of individual galaxies is otherwise not observable on human timescales. I have a wide variety of research interests in this area spanning observations, theory and data science.


## 3D shapes of protogalaxies 
As a core member of the [Cosmic Evolution Early Release Science (CEERS) Survey](https://ceers.github.io/), with [Dr. Haowen Zhang](https://zhw11387.wixsite.com/mysite) and others, I used the James Webb Space Telescope to show that most galaxies in the early Universe are preferentially prolate, not disks as commonly assumed. This has dramatic implications for astrophysics and possibly offers a new tracer or nuisance for cosmology. Figure 5 of  [Pandya+24](https://ui.adsabs.harvard.edu/abs/2024ApJ...963...54P/abstract) illustrates different 3D shapes and their joint distribution of projected sizes and axis ratios. With [Marina Dunn](https://marinadunn.github.io/) and [Dr. Marc Huertas-Company](https://mhuertascompany.github.io/) we are extending this to Euclid. With my student [Arnav Shah](https://www.linkedin.com/in/arnav-shah-ut123), [Prof. Steve Finkelstein](https://www.as.utexas.edu/~stevenf/), and [Prof. Raymond Simons](https://academic-affairs.providence.edu/raymond-simons-ph-d/) we explored [gas kinematics as a tracer for 3D shape](https://ui.adsabs.harvard.edu/abs/2025AAS...24546605S/abstract).

<img width="783" height="835" alt="image" src="https://github.com/user-attachments/assets/4d88830d-43ae-467a-a866-d16a4b6fb796" />


## Intrinsic alignments of prolate protogalaxies
I originally got interested in prolate galaxies because early during my PhD with Profs. [Joel Primack](https://primack.sites.ucsc.edu/), [Avishai Dekel](https://phys.huji.ac.il/people/avishai-dekel), [Sandy Faber](https://www.ucolick.org/~faber/), and [David Koo](https://www.ucolick.org/~koo/), we proposed that these galaxies may align with each other on large scales, tracing out dark matter filaments of the cosmic web. Figure 1 of [Pandya+19](https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.5580P/abstract)  illustrates such alignments, which could be a new cosmological probe of the collapse of dark matter filaments in the early Universe. With my collaborators [Dr. Farhanul Hasan](https://www.stsci.edu/stsci-research/research-directory/farhanul-hasan) and [Dr. Haowen Zhang](https://zhw11387.wixsite.com/mysite) and others, we showed that the Roman Space Telescope can help us study these alignments.

<img width="450" height="545" alt="image" src="https://github.com/user-attachments/assets/df914a46-d5bb-462e-a67a-2a22b97f8fde" />


## Weak gravitational lensing
My interests in intrinsic alignments and 3D shapes of early Universe galaxies led to a collaboration with [Prof. Avi Loeb](https://astronomy.fas.harvard.edu/people/avi-loeb) on the implications for weak gravitational lensing, which is a major effort in precision cosmology. Using JWST-CEERS as one example "blank field" we found evidence for correlated ellipticities and orientations suggesting the presence of dark overdensities in the foreground, though we could not rule out intrinsic alignments. Figure 12 of [Pandya+25](https://ui.adsabs.harvard.edu/abs/2025ApJ...986...72P/abstract) shows polarization-like patterns in the large-scale orientations of galaxies in some sub-regions of the survey.

<img width="941" height="756" alt="image" src="https://github.com/user-attachments/assets/021c2eb6-6de7-4c3f-8a2d-da2da343394f" />


## Star formation and morphological evolution
Like stars, galaxies follow tight "scaling relations" and one of the most famous is the star-forming main sequence: the rate at which they form new stars vs. their mass in existing stars. This is one of numerous observational examples suggesting that galaxies self-regulate their star formation and morphological evolution through feedback loops that dynamical systems approaches can help us understand. Figure 5 from [Pandya+17a](https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.2054P/abstract) illustrates various trajectories that galaxies can take to oscillate on or depart from this main sequence, using models from [Dr. Rachel Somerville](https://www.simonsfoundation.org/people/rachel-somerville/) and [Dr. Ena Choi](https://sites.google.com/site/astroenachoi/ena-choi-astrophysicist).


<img width="740" height="767" alt="image" src="https://github.com/user-attachments/assets/5e163ce8-ccde-4822-8191-9f9dd5b09c47" />


## Interstellar gas in local giant ellipticals
As part of the [MASSIVE Survey](https://blackhole.berkeley.edu/) with [Prof. Jenny Greene](https://web.astro.princeton.edu/people/jenny-greene) I performed emission line spectroscopy to find substantial reservoirs of warm ionized interstellar gas in local giant ellipticals. Figure 2 of [Pandya+17b](https://ui.adsabs.harvard.edu/abs/2017ApJ...837...40P/abstract) shows one such example which is surprising because the galaxy is otherwise "red and dead" and not forming many, if any, new stars.


<img width="947" height="322" alt="image" src="https://github.com/user-attachments/assets/d6ec654a-55a3-47e1-b8b4-a1e549ad847e" />


## Ancient stars in ultra-diffuse galaxies

I performed the first spectral energy distribution (SED) fitting to understand the stellar populations and formation histories of ultra-diffuse galaxies, combining ground-based optical imaging and space-based Spitzer IR imaging. This work was done with [Dr. Aaron Romanowsky](https://www.sjsu.edu/people/aaron.romanowsky/), [Dr. Jean Brodie](https://experts.swinburne.edu.au/5840-jean-brodie) and the SAGES team at UC Santa Cruz. Figure 3 from [Pandya+18](https://ui.adsabs.harvard.edu/abs/2018ApJ...858...29P/abstract) shows some example posterior predictive checks using a very early version of the now popular [prospector](https://github.com/bd-j/prospector) Bayesian code.

<img width="955" height="319" alt="image" src="https://github.com/user-attachments/assets/703b32ed-2476-486e-998d-748e9c9f27d8" />


## Intermediate-mass black holes in ultra-compact dwarf galaxies 

Besides studying the biggest and fluffiest galaxies above, I have also searched for intermediate-mass black holes in the tiniest galaxies using X-ray and radio observations with [Dr. John Mulchaey](https://carnegiescience.edu/dr-john-mulchaey) and [Prof. Jenny Greene](https://web.astro.princeton.edu/people/jenny-greene). Figure 1 from [Pandya+16](https://ui.adsabs.harvard.edu/abs/2016ApJ...819..162P/abstract) shows faint X-rays from several ultra-compact dwarf galaxies. 

<img width="885" height="386" alt="image" src="https://github.com/user-attachments/assets/d7ed5429-486d-448b-8759-b764ecd6bca4" />

# Miscellaneous Projects

One of the beautiful things about galaxy formation is that it involves pretty much every subfield of astrophysics and cosmology, and has deep connections to dynamics, numerics, statistics and interpretable AI/ML. I chose to work in this field because it meant that I had to pursue doctoral and post-doctoral training as an interdisciplinary generalist. I like to stay grounded about how much we still don't know about the various constituents of galaxies by collaborating across research domains. Here are some example projects:


