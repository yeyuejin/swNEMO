MODULE test_arrays_mod
USE obs_fbm

INTERFACE test_arrays
	MODULE PROCEDURE test_real_arrays
	MODULE PROCEDURE test_real_arrays_2D
	MODULE PROCEDURE test_real_arrays_3D
	MODULE PROCEDURE test_integer_arrays
END INTERFACE test_arrays

CONTAINS

   LOGICAL FUNCTION test_real_arrays(array_in,array_out)
	IMPLICIT NONE
	INTEGER :: i
	REAL(KIND=fbdp) :: array_in(:), array_out(:)

	test_real_arrays=.TRUE.
	DO i=1,SIZE(array_in)
		IF (array_in(i) /= array_out(i)) THEN
			test_real_arrays=.FALSE.
		END IF
	END DO

	IF (SIZE(array_in) /= SIZE(array_out)) THEN
			test_real_arrays=.FALSE.
	END IF

	END FUNCTION test_real_arrays


   LOGICAL FUNCTION test_real_arrays_2D(array_in,array_out)
	IMPLICIT NONE
	INTEGER :: i, j
	REAL(KIND=fbdp) :: array_in(:,:), array_out(:,:)

	test_real_arrays_2D=.TRUE.
	DO j=1,SIZE(array_in,2)
		DO i=1,SIZE(array_in,1)
			IF (array_in(i,j) /= array_out(i,j)) THEN
				test_real_arrays_2D=.FALSE.
			END IF
		END DO
	END DO

	IF (SIZE(array_in) /= SIZE(array_out)) THEN
			test_real_arrays_2D=.FALSE.
	END IF

	END FUNCTION test_real_arrays_2D


   LOGICAL FUNCTION test_real_arrays_3D(array_in,array_out)
	IMPLICIT NONE
	INTEGER :: i, j, k
	REAL(KIND=fbdp) :: array_in(:,:,:), array_out(:,:,:)

	test_real_arrays_3D=.TRUE.
	DO k=1,SIZE(array_in,3)
		DO j=1,SIZE(array_in,2)
			DO i=1,SIZE(array_in,1)
				IF (array_in(i,j,k) /= array_out(i,j,k)) THEN
					test_real_arrays_3D=.FALSE.
				END IF
			END DO
		END DO
	END DO

	IF (SIZE(array_in) /= SIZE(array_out)) THEN
			test_real_arrays_3D=.FALSE.
	END IF

	END FUNCTION test_real_arrays_3D


   LOGICAL FUNCTION test_integer_arrays(array_in,array_out)
	IMPLICIT NONE
	INTEGER :: i
	INTEGER :: array_in(:), array_out(:)

	test_integer_arrays=.TRUE.
	DO i=1,SIZE(array_in)
		IF (array_in(i) /= array_out(i)) THEN
			test_integer_arrays=.FALSE.
		END IF
	END DO

	IF (SIZE(array_in) /= SIZE(array_out)) THEN
			test_integer_arrays=.FALSE.
	END IF

	END FUNCTION test_integer_arrays

END MODULE test_arrays_mod
