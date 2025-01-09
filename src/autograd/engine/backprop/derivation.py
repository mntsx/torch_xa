# python 3.12

from typing import Optional, Tuple, Union


class Partial:
    """Represents a univariate partial derivative component.

    Caches whether one partial depends on another and stores
    precomputed derivatives for reuse.

    Attributes:
        _depends_cache (dict[Tuple[int, int], bool]):
            A cache mapping from (id(partialA), id(partialB)) to whether partialA
            depends on partialB.
        _deriv_cache (dict[Tuple[int, int], "ProductGroup"]):
            A cache mapping from (id(partialA), id(partialB)) to the derivative
            ProductGroup of partialA w.r.t. partialB.
        _name (str): The name of this partial (e.g., "x").
        _order (int): The derivative order of this partial.
        _input (Optional[Partial]): The partial on which this partial depends.
    """

    _depends_cache: dict[Tuple[int, int], bool] = {}
    _deriv_cache: dict[Tuple[int, int], "ProductGroup"] = {}

    def __init__(
        self, name: str, input: Optional["Partial"] = None, order: int = 0
    ) -> None:
        """Initializes a Partial.

        Args:
            name (str): The name of this partial (e.g., "A", "B", "X").
            input (Optional[Partial]): The partial on which this one depends.
            order (int): The derivative order (e.g., 2 means d²/dx²).
        """
        self._name: str = name
        self._order: int = order
        self._input: Optional["Partial"] = input

    def __str__(self) -> str:
        """Returns a string representation of the partial.

        Returns:
            str: A string of the form "name(order)".
        """
        return f"{self._name}({self._order})"

    @property
    def name(self) -> str:
        """str: The name of this partial."""
        return self._name

    @property
    def order(self) -> int:
        """int: The derivative order of this partial."""
        return self._order

    @property
    def input(self) -> Optional["Partial"]:
        """Optional[Partial]: The partial on which this one depends, if any."""
        return self._input

    def depends_on_variable(self, variable: "Partial") -> bool:
        """Checks if this partial depends on the specified variable.

        Args:
            variable (Partial): The variable to check dependency on.

        Returns:
            bool: True if this partial depends on 'variable'.
        """
        cache_key: Tuple[int, int] = (id(self), id(variable))
        if cache_key in Partial._depends_cache:
            return Partial._depends_cache[cache_key]

        depends: bool = self._name == variable.name
        if not depends and self._input is not None:
            depends = self._input.depends_on_variable(variable)

        Partial._depends_cache[cache_key] = depends
        return depends

    def derivate(self, variable: "Partial") -> "ProductGroup":
        """Computes the derivative of this partial with respect to 'variable'.

        Args:
            variable (Partial): The variable with respect to which we derive.

        Returns:
            ProductGroup: The derivative of this partial.
        """
        cache_key: Tuple[int, int] = (id(self), id(variable))
        if cache_key in Partial._deriv_cache:
            return Partial._deriv_cache[cache_key]

        derivative: "ProductGroup" = ProductGroup()

        # If we do depend on the variable but are not exactly it, we increment the order
        if self.depends_on_variable(variable) and self._name != variable.name:
            partial: Partial = Partial(
                name=self._name, input=self._input, order=self._order + 1
            )
            derivative.extend(ProductGroup(partial))

            # Chain rule if needed
            if self._input is not None and self._input.name != variable.name:
                derivative.extend(self._input.derivate(variable))

        Partial._deriv_cache[cache_key] = derivative
        return derivative


class ProductGroup:
    """Represents a product of Partial objects multiplied together.

    Attributes:
        _partials (list[Partial]): The list of partials that make up this product.
        _coefficient (int): An integer factor by which the product is multiplied.
        _needs_sort (bool): A flag indicating whether the product needs sorting.
    """

    def __init__(self, partial: Optional[Partial] = None) -> None:
        """Initializes a ProductGroup.

        Args:
            partial (Optional[Partial]): A single partial to start the product with.
                If None, creates an empty product.
        """
        if partial is None:
            self._partials: list[Partial] = []
        else:
            self._partials: list[Partial] = [partial]

        self._coefficient: int = 1
        self._needs_sort: bool = True

    def __len__(self) -> int:
        """Returns the number of Partial objects in this product.

        Returns:
            int: The count of partials in this group.
        """
        return len(self._partials)

    def __str__(self) -> str:
        """Returns the string representation of this product group.

        The partials are sorted, and if _coefficient != 1,
        the string is prefixed by `coefficient*`.

        Returns:
            str: The product's string representation, e.g. `2*A(1) B(2)`.
        """
        self.ensure_sorted()
        coeff_str: str = f"{self._coefficient}*" if self._coefficient != 1 else ""
        partials_str: str = " ".join(str(p) for p in self._partials)
        return coeff_str + partials_str

    @property
    def partials(self) -> list[Partial]:
        """list[Partial]: The partials in this product group."""
        return self._partials

    @property
    def coefficient(self) -> int:
        """int: The integer coefficient of this product."""
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value: int) -> None:
        """Sets the product's integer coefficient.

        Args:
            value (int): The new coefficient.
        """
        self._coefficient = value

    @property
    def id(self) -> Tuple[int, ...]:
        """
        Tuple[int, ...]: A tuple of the partials' orders, used for grouping or hashing.
        """
        self.ensure_sorted()
        return tuple(p.order for p in self._partials)

    def _key(self, partial: Partial) -> Tuple[int, int]:
        """Computes the sorting key for a single partial.

        Args:
            partial (Partial): The partial whose sorting key is needed.

        Returns:
            Tuple[int, int]: The sorting key (-dependencies, -order).
        """
        order: int = partial.order
        deps: int = 0
        current: Optional[Partial] = partial.input
        while current is not None:
            deps += 1
            current = current.input
        return (-deps, -order)

    def ensure_sorted(self) -> None:
        """Sorts the partials by the custom key if needed."""
        if self._needs_sort and len(self._partials) > 1:
            self._partials.sort(key=self._key)
        self._needs_sort = False

    def extend(self, group: Union["ProductGroup", Partial]) -> None:
        """Extends this product by either a single Partial or another ProductGroup.

        Args:
            group (Union[ProductGroup, Partial]): The Partial or ProductGroup to add.

        Raises:
            TypeError: If 'group' is neither a ProductGroup nor a Partial.
        """
        if isinstance(group, ProductGroup):
            self._partials.extend(group.partials)
        elif isinstance(group, Partial):
            self._partials.append(group)
        else:
            raise TypeError("extend() accepts a ProductGroup or Partial.")
        self._needs_sort = True

    def distribute(self, group: "SumGroup") -> "SumGroup":
        """Distributes this product over a SumGroup.

        This is used for the (p1 + p2 + ... ) * this_product operation,
        purely combining partials (no coefficient multiplication here).

        Args:
            group (SumGroup): The sum group to be distributed over.

        Returns:
            SumGroup: A new sum group containing the distributed products.
        """
        self.ensure_sorted()
        sum_group: "SumGroup" = SumGroup()
        for term in group.products:
            product: "ProductGroup" = ProductGroup()
            # combine partials from self
            for p in self._partials:
                product.extend(p)
            # combine partials from the term
            product.extend(term)
            sum_group.extend(SumGroup(product=product))
        return sum_group

    def derivate(self, variable: Partial) -> "SumGroup":
        """Derives this product group with respect to a variable using the product rule.

        Args:
            variable (Partial): The variable with respect to which we derive.

        Returns:
            SumGroup: The derivative of this product group, stored as a sum of products.
        """
        length: int = len(self._partials)
        self.ensure_sorted()

        # derivative of empty => 0
        if length == 0:
            return SumGroup()

        # single partial
        if length == 1:
            single_product: "ProductGroup" = ProductGroup()
            single_product.coefficient = self._coefficient
            single_partial: Partial = self._partials[0]
            single_product.extend(single_partial.derivate(variable))
            return SumGroup(product=single_product)

        # two-factor product rule
        if length == 2:
            sum_group: "SumGroup" = SumGroup()
            p1: Partial = self._partials[0]
            p2: Partial = self._partials[1]

            productA: "ProductGroup" = ProductGroup(p2)
            productA.extend(p1.derivate(variable))
            productA.coefficient = self._coefficient

            productB: "ProductGroup" = ProductGroup(p1)
            productB.extend(p2.derivate(variable))
            productB.coefficient = self._coefficient

            sum_group.extend(SumGroup(productA))
            sum_group.extend(SumGroup(productB))
            return sum_group

        # general case (3+ factors)
        sum_group: "SumGroup" = SumGroup()

        p0: Partial = self._partials[0]
        rest_group: "ProductGroup" = ProductGroup()
        for partial in self._partials[1:]:
            rest_group.extend(partial)

        # part A: derivative of p0 * rest
        productA: "ProductGroup" = ProductGroup()
        productA.extend(p0.derivate(variable))
        productA.extend(rest_group)
        productA.coefficient = self._coefficient

        # part B: p0 * derivative of rest
        sum_group_B: "SumGroup" = ProductGroup(p0).distribute(
            rest_group.derivate(variable)
        )
        for pgr in sum_group_B.products:
            pgr.coefficient *= self._coefficient

        sum_group.extend(SumGroup(product=productA))
        sum_group.extend(sum_group_B)
        return sum_group


class SumGroup:
    """Represents a sum of ProductGroups.

    Each ProductGroup can be combined (grouped) if they share the same
    sorted partial ID. The final result is printed as a sum.

    Attributes:
        _products (list[ProductGroup]): The list of product groups in the sum.
        _needs_sort (bool): Flag indicating whether we need to sort the product list.
    """

    def __init__(self, product: Optional["ProductGroup"] = None) -> None:
        """Initializes a SumGroup.

        Args:
            product (Optional[ProductGroup]): A single product group to start with.
        """
        if product is None:
            self._products: list[ProductGroup] = []
        else:
            self._products: list[ProductGroup] = [product]

        self._needs_sort: bool = True

    def __len__(self) -> int:
        """Returns the number of product groups in this sum.

        Returns:
            int: The number of product groups.
        """
        return len(self._products)

    def __str__(self) -> str:
        """Creates a string representation of the sum group.

        Each product group is displayed with bracket notation if its coefficient != 1,
        e.g.: `3[ A(2) B(1) ]`.

        Returns:
            str: The string representation of this sum group.
        """
        self.ensure_sorted()
        parts: list[str] = []
        for product in self._products:
            product.ensure_sorted()
            partials_str: str = " ".join(str(p) for p in product.partials)
            coeff: int = product.coefficient

            if coeff == 1:
                # If coefficient is 1, just show the partials
                parts.append(partials_str)
            else:
                # e.g., "3[ A(2) B(1) ]"
                parts.append(f"{coeff}[ {partials_str} ]")

        return " + ".join(parts)

    def group(self) -> None:
        """Merges products with identical IDs by summing their coefficients."""
        self.ensure_sorted()
        merged: dict[Tuple[int, ...], ProductGroup] = {}
        for product in self._products:
            pid: Tuple[int, ...] = product.id
            if pid not in merged:
                merged[pid] = product
            else:
                merged[pid].coefficient += product.coefficient

        self._products = list(merged.values())
        self._needs_sort = True

    @property
    def products(self) -> list["ProductGroup"]:
        """list[ProductGroup]: The list of product groups."""
        return self._products

    def _key(self, product: "ProductGroup") -> int:
        """Computes the sort key for a ProductGroup based on the sum of orders.

        Args:
            product (ProductGroup): The product to sort.

        Returns:
            int: A negative sum of partial orders (for descending sort).
        """
        orders: Tuple[int, ...] = product.id
        return -sum(orders) if orders else 0

    def ensure_sorted(self) -> None:
        """Sorts the product groups by the custom key if needed."""
        if self._needs_sort and len(self._products) > 1:
            self._products.sort(key=self._key)
        self._needs_sort = False

    def extend(self, group: "SumGroup") -> None:
        """Adds another sum group's products to this one.

        Args:
            group (SumGroup): The sum group to add.
        """
        self._products.extend(group.products)
        self._needs_sort = True

    def derivate(self, variable: Partial) -> "SumGroup":
        """Computes the derivative of this sum group w.r.t. the given variable.

        Steps:
          1. group() -> merges duplicates
          2. Derive each product
          3. group() -> merges duplicates again

        Args:
            variable (Partial): The variable with respect to which we derive.

        Returns:
            SumGroup: The derivative of this sum group.
        """
        self.group()
        derivative: "SumGroup" = SumGroup()
        for product in self._products:
            part_deriv: "SumGroup" = product.derivate(variable)
            derivative.extend(part_deriv)
        derivative.group()
        return derivative


def calculate_n_order_partial(n: int) -> "SumGroup":
    """Example method to compute an nth-order partial derivative.

    Creates a dependency chain A->B->X, then takes the nth derivative
    w.r.t. X.

    Args:
        n (int): The order of the derivative to compute.

    Returns:
        SumGroup: The resulting SumGroup after taking n derivatives.
    """
    X: Partial = Partial(name="X", input=None, order=0)
    B: Partial = Partial(name="B", input=X, order=0)
    A: Partial = Partial(name="A", input=B, order=0)

    sum_group: "SumGroup" = SumGroup(product=ProductGroup(A))

    for _ in range(n):
        sum_group = sum_group.derivate(X)

    return sum_group
