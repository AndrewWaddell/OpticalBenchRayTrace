p = [...
    0 0 0;... % 1
    1 0 0;... % 2
    0 1 0;... % 3
    1 1 0;... % 4
    0 0 1;... % 5
    1 0 1;... % 6
    0 1 1;... % 7
    1 1 1;... % 8
    ];
cm = [...
    1 2 3;...
    2 3 4;...
    1 2 5;...
    2 5 6;...
    1 3 5;...
    3 5 7;...
    2 4 6;...
    4 6 8;...
    3 4 7;...
    4 7 8;...
    5 6 7;...
    6 7 8;...
    ];

TR = triangulation(cm,p);
% trisurf(TR)
stlwrite(TR,'test_cube.stl')


