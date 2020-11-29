include "globals.mzn";

function var bool : assignment(array[int] of var int: x,
			     array[int] of int: input_x) =
	forall( i in index_set(x))(x[i] == input_x[i]);

function var bool: rooms_equal(var SIDES: r1_tx,var SIDES: r1_ty,
                           var SIDES: r1_bx, var SIDES: r1_by,
		                   var SIDES: r2_tx, var SIDES: r2_ty,
		                   var SIDES: r2_bx, var SIDES: r2_by) =
		(r1_tx == r2_tx) /\ (r1_ty == r2_ty) /\
		(r1_bx == r2_bx) /\ (r1_by == r2_by);

function var bool:
        horizontal_overlap(var SIDES: r1_tx, var SIDES: r1_bx,
                           var SIDES: r2_tx, var SIDES: r2_bx) =
            (r1_tx > r2_bx) /\ (r1_bx < r2_tx);


function var bool:
        vertical_overlap(var SIDES: r1_ty, var SIDES: r1_by,
                         var SIDES: r2_ty, var SIDES: r2_by) =
            (r1_ty < r2_by) /\ (r1_by > r2_ty);



function var bool:
        horizontal_overlap_corridor(var SIDES: r1_tx, var SIDES: r1_bx,
                           var SIDES: r2_tx, var SIDES: r2_bx) =
            (r1_tx > r2_bx) /\ (r1_bx < r2_tx) /\ (r1_tx - r1_bx == 1);


function var bool:
        vertical_overlap_corridor(var SIDES: r1_ty, var SIDES: r1_by,
                         var SIDES: r2_ty, var SIDES: r2_by) =
            (r1_ty < r2_by) /\ (r1_by > r2_ty) /\ (r1_ty - r1_by == 1);


function var bool: overlap(var SIDES: r1_tx,var SIDES: r1_ty,
                           var SIDES: r1_bx,var SIDES: r1_by,
		                   var SIDES: r2_tx,var SIDES: r2_ty,
		                   var SIDES: r2_bx, var SIDES: r2_by) =

        horizontal_overlap(r1_tx, r1_bx, r2_tx, r2_bx) /\
        vertical_overlap(r1_ty, r1_by, r2_ty, r2_by);


function var bool: adj_x(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by,
		               var SIDES: r2_tx, var SIDES: r2_ty,
		               var SIDES: r2_bx, var SIDES: r2_by) =

	     (r1_tx==r2_bx \/ r1_bx == r2_tx) /\

	     vertical_overlap(r1_ty, r1_by, r2_ty, r2_by);


function var bool: adj_y(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by,
		               var SIDES: r2_tx, var SIDES: r2_ty,
		               var SIDES: r2_bx, var SIDES: r2_by) =

         (r1_ty==r2_by \/ r1_by == r2_ty) /\

	     horizontal_overlap(r1_tx, r1_bx, r2_tx, r2_bx);



function var bool: adj(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by,
		               var SIDES: r2_tx, var SIDES: r2_ty,
		               var SIDES: r2_bx, var SIDES: r2_by) =

        adj_x(r1_tx, r1_ty, r1_bx, r1_by, r2_tx, r2_ty, r2_bx, r2_by) \/
        adj_y(r1_tx, r1_ty, r1_bx, r1_by, r2_tx, r2_ty, r2_bx, r2_by);

function var bool: adj_x_corridor(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by,
		               var SIDES: r2_tx, var SIDES: r2_ty,
		               var SIDES: r2_bx, var SIDES: r2_by) =

	     (r1_tx==r2_bx \/ r1_bx == r2_tx) /\

	     vertical_overlap_corridor(r1_ty, r1_by, r2_ty, r2_by);


function var bool: adj_y_corridor(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by,
		               var SIDES: r2_tx, var SIDES: r2_ty,
		               var SIDES: r2_bx, var SIDES: r2_by) =

         (r1_ty==r2_by \/ r1_by == r2_ty) /\

	     horizontal_overlap_corridor(r1_tx, r1_bx, r2_tx, r2_bx);

function var bool: adj_corridor(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by,
		               var SIDES: r2_tx, var SIDES: r2_ty,
		               var SIDES: r2_bx, var SIDES: r2_by) =

        adj_x_corridor(r1_tx, r1_ty, r1_bx, r1_by, r2_tx, r2_ty, r2_bx, r2_by) \/
        adj_y_corridor(r1_tx, r1_ty, r1_bx, r1_by, r2_tx, r2_ty, r2_bx, r2_by);

function var int: area(var SIDES: r1_tx, var SIDES: r1_ty,
                       var SIDES: r1_bx, var SIDES: r1_by) =
        (r1_tx - r1_bx) * (r1_by - r1_ty);

function var bool: door_on_wall(var SIDES: r1_tx, var SIDES: r1_ty,
                                var SIDES: r1_bx, var SIDES: r1_by,
                                 var int: SIDE) =

        adj(r1_tx, r1_ty, r1_bx, r1_by, SIDE-1, 0, 1, 0) \/
        adj(r1_tx, r1_ty, r1_bx, r1_by, 0, 1, 0, SIDE-1) \/
        adj(r1_tx, r1_ty, r1_bx, r1_by, SIDE, 1, SIDE, SIDE-1) \/
        adj(r1_tx, r1_ty, r1_bx, r1_by, 1, SIDE, SIDE-1, SIDE);

int: SIDE;
constraint assert(SIDE>0, "error, the side of the apartment must be greater than 0");
set of int: SIDES = 0..SIDE;
int: ROOM_NUMBER;
set of int: ROOMS = 1..ROOM_NUMBER;
int: WALL_CELLS;
set of int: CELLS = 1..WALL_CELLS;

%% ROOM TYPES PARAMETERS %%
int: ROOM_TYPES_N = 8;
int: COMMON_ROOM = 1;
int: KITCHEN = 2;
int: BEDROOM = 3;
int: BATHROOM = 4;
int: CORRIDOR = 5;
int: LIVING_ROOM = 6;
int: LIVING_AREA = 7;
int: RESTING_AREA = 8;

var SIDES: door_bx;
var SIDES: door_by;
var SIDES: door_tx;
var SIDES: door_ty;

%% arrays of lb and ub of number of types of rooms %%
set of int: ROOM_TYPES = 1..ROOM_TYPES_N;

array[ROOM_TYPES] of int: area_rooms;

array[CELLS] of int: wall_bx;
array[CELLS] of int: wall_by;
array[CELLS] of int: wall_tx;
array[CELLS] of int: wall_ty;
var CELLS: door_cell;

constraint ( door_bx == wall_bx[door_cell] /\
             door_by == wall_by[door_cell] /\
             door_tx == wall_tx[door_cell] /\
             door_ty == wall_ty[door_cell]);


array[ROOMS] of var SIDES: rooms_bx;
array[ROOMS] of var SIDES: rooms_by;
array[ROOMS] of var SIDES: rooms_tx;
array[ROOMS] of var SIDES: rooms_ty;
array[ROOMS] of var ROOM_TYPES: rooms_types;
array[ROOMS,ROOMS] of int: rooms_adj;
array[ROOMS] of ROOM_TYPES: input_rooms_types;

constraint forall(s in ROOMS)(
        (rooms_bx[s] < rooms_tx[s]) /\ (rooms_by[s] > rooms_ty[s]));

constraint forall(s in ROOMS)(
        area(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s]) == area_rooms[rooms_types[s]]);

constraint(
    sum(s in ROOMS) (area(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s])) <= SIDE * SIDE);

constraint forall(s,s1 in ROOMS)
     ((s == s1) \/ ( not rooms_equal(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
     rooms_tx[s1], rooms_ty[s1], rooms_bx[s1], rooms_by[s1])));


%different rooms must not overlap%
constraint forall(s, s1 in ROOMS) (

     (s == s1) \/

     (not rooms_equal(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
     rooms_tx[s1], rooms_ty[s1], rooms_bx[s1], rooms_by[s1])

     ->
     not overlap(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
     rooms_tx[s1], rooms_ty[s1], rooms_bx[s1], rooms_by[s1])
     ));


constraint forall(s,s1 in ROOMS)(
    (s == s1) \/

    (rooms_adj[s,s1] == 1 <->
        adj(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
        rooms_tx[s1], rooms_ty[s1], rooms_bx[s1], rooms_by[s1])
     ));


constraint forall(s,s1 in ROOMS)(
    (s == s1) \/

    (rooms_adj[s,s1] == 1 /\ rooms_types[s] == CORRIDOR <->
        adj_corridor(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
        rooms_tx[s1], rooms_ty[s1], rooms_bx[s1], rooms_by[s1])
     ));

constraint forall(s,s1 in ROOMS)(
    (s == s1) \/

    (rooms_adj[s,s1] == 0 <->
        not adj(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
        rooms_tx[s1], rooms_ty[s1], rooms_bx[s1], rooms_by[s1])
     ));

%% constraint ( %%
%%    door_on_wall(door_tx, door_ty, door_bx, door_by, SIDE)); %%

constraint (
    exists(s in ROOMS where rooms_types[s] == COMMON_ROOM)(
                overlap(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
                door_tx, door_ty, door_bx, door_by)));

constraint forall(s in ROOMS)(
    rooms_types[s] != COMMON_ROOM -> (
        not adj(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s],
                door_tx, door_ty, door_bx, door_by)
    )
);




constraint forall(s in ROOMS)(
    rooms_types[s] == CORRIDOR ->(
        ((rooms_tx[s] - rooms_bx[s]) == 1)
            \/
        ((rooms_by[s] - rooms_ty[s]) == 1)
    )
);

%%constraint ( %%
%%        ((door_tx - door_bx) == 1) %%
%%            /\ %%
%%        ((door_by - door_ty) == 1) %%
%% ); %%

constraint forall(s in ROOMS)(
    rooms_types[s] != CORRIDOR /\ area_rooms[rooms_types[s]] > 3

    ->(
        ((rooms_tx[s] - rooms_bx[s]) != 1)
            /\
        ((rooms_by[s] - rooms_ty[s]) != 1)
    )
);

%% constraint forall(s in ROOMS)( %%
%%    rooms_types[s] != CORRIDOR /\ area_rooms[rooms_types[s]] != 1 %%
%% %%
%%    ->( %%
%%        ((rooms_tx[s] - rooms_bx[s]) != 1) %%
%%            /\ %%
%%        ((rooms_by[s] - rooms_ty[s]) != 1) %%
%%    ) %%
%% );


constraint assignment(rooms_types, input_rooms_types);

%%solve maximize sum (s in ROOMS) (area(rooms_tx[s], rooms_ty[s], rooms_bx[s], rooms_by[s]));%%

solve satisfy;

